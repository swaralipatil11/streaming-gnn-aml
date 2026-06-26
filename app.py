import os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager

# Define Pydantic models for endpoints

class AnomalyRequest(BaseModel):
    x: List[List[float]]
    edge_index: List[List[int]]

class AnomalyResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]

class Transaction(BaseModel):
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount: float
    payment_format: str

class TransactionBatchRequest(BaseModel):
    transactions: List[Transaction]

class TransactionBatchResponse(BaseModel):
    predictions: Dict[str, int]
    probabilities: Dict[str, float]

# Global variables for the model
model = None
in_channels = 5
hidden_channels = 64
node_to_idx = {}
node_stats = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that loads the GNN model weights
    from disk during startup.
    """
    global model, node_to_idx, node_stats, in_channels, hidden_channels
    checkpoint_path = "aml_gcn_model.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained model weights from '{checkpoint_path}'...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            in_channels = checkpoint['in_channels']
            hidden_channels = checkpoint['hidden_channels']
            node_to_idx = checkpoint.get('node_to_idx', {})
            node_stats = checkpoint.get('node_stats', {})
            
            from model import AMLGraphNet
            model = AMLGraphNet(in_channels=in_channels, hidden_channels=hidden_channels)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Model, node mapping, and statistics loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}. Running with uninitialized model.")
            from model import AMLGraphNet
            model = AMLGraphNet(in_channels=in_channels, hidden_channels=hidden_channels)
            model.eval()
    else:
        print(f"WARNING: Checkpoint '{checkpoint_path}' not found. Starting with uninitialized model.")
        from model import AMLGraphNet
        model = AMLGraphNet(in_channels=in_channels, hidden_channels=hidden_channels)
        model.eval()
        
    yield

from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Relational GNN Anomaly & Anti-Money Laundering Detection API",
    description="High-throughput inference API using a 2-hop GCN to classify accounts as Licit or Illicit.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable Cross-Origin Resource Sharing (CORS) for local React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins (e.g. ['http://localhost:5173'])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

@app.post("/predict_anomaly", response_model=AnomalyResponse)
async def predict_anomaly(req: AnomalyRequest):
    """
    Accepts raw tensor features (x) and sparse coordinates (edge_index) of a graph,
    runs the GCN in an evaluation context (torch.no_grad), and returns binary predictions.
    """
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="GNN model is not loaded.")
        
    if not req.x or not req.edge_index:
        raise HTTPException(status_code=400, detail="x and edge_index cannot be empty.")

    try:
        x_tensor = torch.tensor(req.x, dtype=torch.float32)
        edge_index_tensor = torch.tensor(req.edge_index, dtype=torch.long)
        
        with torch.no_grad():
            out = model(x_tensor, edge_index_tensor)
            probs = torch.exp(out)  # model returns log-softmax, so exp maps it back to [0, 1]
            preds = out.argmax(dim=1)
            
        return AnomalyResponse(
            predictions=preds.tolist(),
            probabilities=probs.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference processing failed: {str(e)}")

@app.post("/predict_transactions", response_model=TransactionBatchResponse)
async def predict_transactions(req: TransactionBatchRequest):
    """
    Accepts a streaming JSON batch of raw transaction structures, dynamically maps
    them to a local subgraph, engineers features using log1p-scaling, and performs GCN inference
    to classify accounts.
    """
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="GNN model is not loaded.")
        
    if not req.transactions:
        return TransactionBatchResponse(predictions={}, probabilities={})

    try:
        # 1. Map account string IDs to a local subgraph index system
        account_set = set()
        for tx in req.transactions:
            src_id = f"{tx.from_bank}_{tx.from_account}"
            dst_id = f"{tx.to_bank}_{tx.to_account}"
            account_set.add(src_id)
            account_set.add(dst_id)
            
        unique_accounts = sorted(list(account_set))
        local_node_to_idx = {acc: i for i, acc in enumerate(unique_accounts)}
        num_nodes = len(unique_accounts)
        
        # 2. Extract subgraph node attributes (degrees, transaction amounts)
        in_degree = [0] * num_nodes
        out_degree = [0] * num_nodes
        amount_sent = [0.0] * num_nodes
        amount_received = [0.0] * num_nodes
        
        edge_sources = []
        edge_targets = []
        
        for tx in req.transactions:
            src_id = f"{tx.from_bank}_{tx.from_account}"
            dst_id = f"{tx.to_bank}_{tx.to_account}"
            
            src_idx = local_node_to_idx[src_id]
            dst_idx = local_node_to_idx[dst_id]
            
            edge_sources.append(src_idx)
            edge_targets.append(dst_idx)
            
            out_degree[src_idx] += 1
            in_degree[dst_idx] += 1
            amount_sent[src_idx] += tx.amount
            amount_received[dst_idx] += tx.amount
            
        # 3. Create node features (scaled with log1p for stability)
        # We combine batch stats with global historical stats loaded from the checkpoint to prevent feature shift
        x_features = []
        for idx in range(num_nodes):
            acc_id = unique_accounts[idx]
            # Lookup historical stats (default to 0 if node is new)
            hist_in_deg, hist_out_deg, hist_sent, hist_recv = node_stats.get(acc_id, [0, 0, 0.0, 0.0])
            
            in_deg = hist_in_deg + in_degree[idx]
            out_deg = hist_out_deg + out_degree[idx]
            sent = hist_sent + amount_sent[idx]
            recv = hist_recv + amount_received[idx]
            
            x_features.append([
                np.log1p(in_deg),
                np.log1p(out_deg),
                np.log1p(sent),
                np.log1p(recv),
                np.log1p(in_deg + out_deg)
            ])
            
        x_tensor = torch.tensor(x_features, dtype=torch.float32)
        edge_index_tensor = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        
        # 4. GCN inference execution
        with torch.no_grad():
            out = model(x_tensor, edge_index_tensor)
            probs = torch.exp(out)
            preds = out.argmax(dim=1)
            
        predictions_dict = {}
        probabilities_dict = {}
        
        for i, acc in enumerate(unique_accounts):
            predictions_dict[acc] = int(preds[i].item())
            # Index 1 corresponds to 'Illicit' node class probability
            probabilities_dict[acc] = float(probs[i, 1].item())
            
        return TransactionBatchResponse(
            predictions=predictions_dict,
            probabilities=probabilities_dict
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transaction batch inference failed: {str(e)}")

# Serve static React frontend files from root if they are built
dist_dir = "frontend/dist"
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        """Fallback API status check when frontend build is not found."""
        return {
            "status": "online",
            "model_loaded": model is not None,
            "nodes_registered_in_registry": len(node_to_idx),
            "message": "React frontend built folder not found. Run 'npm run build' inside the 'frontend' folder to serve the visual dashboard."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
