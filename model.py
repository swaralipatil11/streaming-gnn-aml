import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class AMLGraphNet(torch.nn.Module):
    """
    AMLGraphNet is a Graph Convolutional Network (GCN) designed for
    anti-money laundering (AML) detection. It aggregates transactional context
    across a 2-hop neighborhood (multi-layer message passing) to detect
    anomalous behavior in account nodes.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 2, dropout: float = 0.3):
        super(AMLGraphNet, self).__init__()
        
        # First GCN layer (1-hop aggregation)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Second GCN layer (2-hop aggregation)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Fully connected classification head
        self.fc = nn.Linear(hidden_channels, out_channels)
        
        # Dropout probability
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the GCN model.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Graph connectivity matrix of shape [2, num_edges]
            edge_weight: Optional 1D edge weights of shape [num_edges]
            
        Returns:
            Log-softmax probabilities of shape [num_nodes, out_channels]
        """
        # Step 1: First Hop message passing + activation + dropout
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Step 2: Second Hop message passing + activation + dropout
        x = self.conv2(x, edge_index, edge_weight)
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
    model = AMLGraphNet(in_channels, hidden_channels)
    
    # Mock data: 10 nodes, 4 edges
    x_mock = torch.rand(10, in_channels)
    edge_index_mock = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    output = model(x_mock, edge_index_mock)
    print("Model Architecture:")
    print(model)
    print("\nForward pass output shape:", output.shape)
    print("Sum of probabilities per node (should be close to 0 in log space since they sum to 1):")
    print(torch.exp(output).sum(dim=1))
