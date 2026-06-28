import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class AMLGraphNet(torch.nn.Module):
    """
    AMLGraphNet is a Graph Attention Network (GAT) designed for
    anti-money laundering (AML) detection. It aggregates transactional context
    across a 2-hop neighborhood (multi-layer message passing) to detect
    anomalous behavior in account nodes, using both node and edge attributes (amount & payment format).
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2, dropout: float = 0.3, edge_dim: int = 9):
        super(AMLGraphNet, self).__init__()
        
        # First GAT layer (1-hop aggregation) with 8 attention heads
        # Output dimension: hidden_channels * 8
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True, edge_dim=edge_dim)
        
        # Second GAT layer (2-hop aggregation) with 1 attention head
        # Output dimension: hidden_channels
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=1, concat=False, edge_dim=edge_dim)
        
        # Fully connected classification head
        self.fc = nn.Linear(hidden_channels, out_channels)
        
        # Dropout probability
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the GAT model.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity matrix of shape [2, num_edges]
            edge_attr: Edge attributes of shape [num_edges, edge_dim]
            
        Returns:
            Log-softmax probabilities of shape [num_nodes, out_channels]
        """
        # Step 1: First Hop message passing + activation + dropout
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Step 2: Second Hop message passing + activation + dropout
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Step 3: Classification projection
        logits = self.fc(x)
        
        # Step 4: Log-softmax for classification optimization using NLL loss
        return F.log_softmax(logits, dim=1)

if __name__ == "__main__":
    # Test model instantiation and forward pass with dummy tensor shapes
    in_channels = 5
    hidden_channels = 16
    edge_dim = 9
    model = AMLGraphNet(in_channels, hidden_channels, edge_dim=edge_dim)
    
    # Mock data: 10 nodes, 4 edges
    x_mock = torch.rand(10, in_channels)
    edge_index_mock = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr_mock = torch.rand(4, edge_dim)
    
    output = model(x_mock, edge_index_mock, edge_attr_mock)
    print("Model Architecture:")
    print(model)
    print("\nForward pass output shape:", output.shape)
    print("Sum of probabilities per node (should be close to 0 in log space since they sum to 1):")
    print(torch.exp(output).sum(dim=1))
