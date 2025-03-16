import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define the Graph Autoencoder Model
class GAEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GAEModel, self).__init__()
        # Encoder: Two GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
    
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        return z
    
    def decode(self, z):
        # Inner product decoder
        adj_reconstructed = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_reconstructed
    
    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        adj_reconstructed = self.decode(z)
        return adj_reconstructed

