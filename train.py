import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

from dataset import StreamGraphBuilder
from model import AMLGraphNet

def train_model(args):
    # 1. Load data
    builder = StreamGraphBuilder(args.data_path, chunksize=args.chunk_size)
    data = builder.build_graph(max_rows=args.max_rows)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    num_nodes = data.x.size(0)
    in_channels = data.x.size(1)
    
    # 2. Split graph nodes into train/test masks (80/20)
    # To be extra rigorous and ensure reproducible splits
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * 0.8)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    # Move graph data to target device
    data = data.to(device)
    
    # 3. Calculate class weights for NLL Loss (Inverse Frequency)
    y_train = data.y[data.train_mask]
    num_licit = (y_train == 0).sum().item()
    num_illicit = (y_train == 1).sum().item()
    total_train = num_licit + num_illicit
    
    if num_illicit == 0:
        print("WARNING: Zero illicit nodes in training mask. Applying default weights.")
        class_weights = torch.tensor([1.0, 10.0], dtype=torch.float, device=device)
    else:
        weight_licit = total_train / (2.0 * max(num_licit, 1))
        weight_illicit = total_train / (2.0 * max(num_illicit, 1))
        class_weights = torch.tensor([weight_licit, weight_illicit], dtype=torch.float, device=device)
        
    print(f"Training Class Counts: Licit={num_licit}, Illicit={num_illicit}")
    print(f"Computed Class Weights: Licit={class_weights[0].item():.4f}, Illicit={class_weights[1].item():.4f}")
    
    # 4. Instantiate model & optimizer
    model = AMLGraphNet(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        dropout=args.dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 5. Training loop
    model.train()
    print("Starting model training...")
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Loss calculation (applying penalized class weights directly in NLL loss)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            # Quick evaluation on training mask
            preds = out[data.train_mask].argmax(dim=1)
            correct = (preds == data.y[data.train_mask]).sum().item()
            train_acc = correct / data.train_mask.sum().item()
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")
            
    # 6. Evaluation strictly using Precision, Recall, and F1-Score via scikit-learn
    print("\nEvaluating model on test set...")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out[data.test_mask].argmax(dim=1).cpu().numpy()
        targets = data.y[data.test_mask].cpu().numpy()
        
    # Calculate performance metrics
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', zero_division=0)
    print("\n=== Test Performance Metrics (Binary: Illicit Class) ===")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("=========================================================")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(targets, preds, target_names=['Licit (0)', 'Illicit (1)'], zero_division=0))
    
    # Construct global node stats map for dynamic inference lookup
    node_stats = {
        node_id: [
            builder.node_in_degree[idx],
            builder.node_out_degree[idx],
            builder.node_amount_sent[idx],
            builder.node_amount_received[idx]
        ] for node_id, idx in builder.node_to_idx.items()
    }
    
    # 7. Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'node_to_idx': builder.node_to_idx,
        'node_stats': node_stats,
        'in_channels': in_channels,
        'hidden_channels': args.hidden_channels,
        'out_channels': 2,
        'dropout': args.dropout,
        'class_weights': class_weights.cpu().numpy().tolist()
    }
    
    save_path = args.save_path
    torch.save(checkpoint, save_path)
    print(f"Model saved successfully to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relational GNN Anomaly/AML Detection Trainer")
    parser.add_argument("--data_path", type=str, default="dataset/HI-Small_Trans.csv", help="Path to raw transaction log CSV")
    parser.add_argument("--chunk_size", type=int, default=200000, help="Pandas reading chunk size")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit number of rows to ingest for quick runs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Dimension of hidden layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_path", type=str, default="aml_gcn_model.pth", help="Filename to save model weights")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    
    args = parser.parse_args()
    train_model(args)
