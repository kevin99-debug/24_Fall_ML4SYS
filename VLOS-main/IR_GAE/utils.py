import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

import pydot
import networkx as nx

# Parse DOT File Function
def parse_dot_file(dot_file_path: str) -> nx.DiGraph:
    graphs = pydot.graph_from_dot_file(dot_file_path)
    if not graphs:
        raise ValueError("No graphs found in DOT file.")
    pydot_graph = graphs[0]
    nx_graph = nx.drawing.nx_pydot.from_pydot(pydot_graph)
    return nx_graph

# Convert NetworkX Graph to PyTorch Geometric Data
def convert_nx_to_pyg_data(nx_graph: nx.DiGraph) -> Data:
    # Map nodes to indices
    node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    nx_graph = nx.relabel_nodes(nx_graph, node_mapping)

    num_nodes = nx_graph.number_of_nodes()

    # Use a constant feature for all nodes
    x = torch.ones((num_nodes, 1), dtype=torch.float)

    # Edge index
    edges = list(nx_graph.edges())
    if len(edges) > 0:
        edge_index = torch.tensor(
            [[source, target] for source, target in edges],
            dtype=torch.long
        ).t().contiguous()
    else:
        # No edges, create an empty edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Add self-loops to ensure edge_index is not empty
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    data = Data(x=x, edge_index=edge_index)
    
    # Create adjacency matrix
    adj = nx.to_scipy_sparse_array(nx_graph)  # Correctly imported function
    adj = torch.tensor(adj.todense(), dtype=torch.float)
    data.adj = adj  # Add adjacency matrix to data object

    return data

# Define the Loss Function
def loss_function(recon_adj, adj_orig):
    # Flatten the matrices
    recon_adj_flat = recon_adj.view(-1)
    adj_orig_flat = adj_orig.view(-1)
    # Compute BCE loss
    loss = F.binary_cross_entropy(recon_adj_flat, adj_orig_flat)
    return loss
