import os
import torch
from model import GAEModel
from preprocess import preprocess
from utils import parse_dot_file, convert_nx_to_pyg_data, loss_function
from datasets import load_dataset  # Ensure this is correctly imported
import time

import csv

# Function to Extract Embeddings
def extract_embedding(model, data, device):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x.to(device), data.edge_index.to(device))
    # Aggregate node embeddings to get a single graph embedding (e.g., mean pooling)
    graph_embedding = z.mean(dim=0)
    return graph_embedding.cpu()

# Function to Load the Model
def load_trained_model(model_path, input_dim, hidden_dim, embedding_dim, device):
    model = GAEModel(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to Generate Embeddings and Save to CSV
def generate_embeddings(dataset, model, device, output_csv, max_samples=None):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['ir_index', 'embedding_vector'])
        sample_count = 0
        for ir_index, module in enumerate(dataset):
            if max_samples is not None and sample_count >= max_samples:
                break
            # if ir_index < 20000:
            #     continue
            try:
                dot_file_path = preprocess(module, ir_index, ir_index)
                nx_graph = parse_dot_file(dot_file_path)
                pyg_data = convert_nx_to_pyg_data(nx_graph)
                embedding = extract_embedding(model, pyg_data, device)
                writer.writerow([ir_index, embedding.tolist()])
                os.remove(dot_file_path)
                # print(f"Processed IR index {ir_index}, Embedding: {embedding_str}")
                sample_count += 1
            except Exception as e:
                print(f"Failed to process IR index {ir_index}, Error: {e}")
                continue
    
    print(f"Embeddings saved to {output_csv}")


# Main Function
def main():
    # Model parameters (ensure these match the training configuration)
    input_dim = 1  # Each node has a single feature value
    hidden_dim = 64
    embedding_dim = 32  # Size of the latent embedding
    output_csv = './embedding_vectors_test.csv'
    max_samples = 1000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model = load_trained_model('./model/final_model.pth', input_dim, hidden_dim, embedding_dim, device)
    print(f"Loaded model")
    
    # Load the dataset used for training
    # Replace 'llvm-ml/ComPile' with your actual dataset name if different
    dataset = load_dataset('llvm-ml/ComPile', split='train', streaming=True)
    print("Loaded dataset for embedding generation.")
    
    # Generate embeddings and save to CSV
    generate_embeddings(dataset, model, device, output_csv, max_samples)

# Run the main function
if __name__ == '__main__':
    main()
