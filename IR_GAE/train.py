import os
import torch
from torch_geometric.loader import DataLoader
from datasets import load_dataset
from model import GAEModel
from preprocess import preprocess
from utils import parse_dot_file, convert_nx_to_pyg_data, loss_function

# Training Function
def train_gnn(model, optimizer, data_loader, loss_fn, device):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_adj = model(data)
        loss = loss_fn(recon_adj, data.adj)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Main Function
def main():
    batch_size = 1  # Process one graph at a time for simplicity
    hidden_dim = 64
    embedding_dim = 32  # Size of the latent embedding
    total_training_data_count = 1500  # Adjust as needed
    epochs = 15000  # Number of training epochs

    input_dim = 1  # Each node has a single feature value

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GAEModel(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = loss_function

    # Initialize training index
    training_index = 0

    # Load the dataset (ensure 'load_dataset' is properly defined)
    dataset = load_dataset('llvm-ml/ComPile', split='train', streaming=True)
    dataset_iter = iter(dataset)

    # Ensure directories exist
    os.makedirs('ir', exist_ok=True)
    os.makedirs('graphs', exist_ok=True)

    for ir_number in range(0, epochs):
        data_batch = []
        count = 0

        while count < batch_size and training_index < total_training_data_count:
            try:
                module = next(dataset_iter)
                dot_file_path = preprocess(module, training_index, count)
                nx_graph = parse_dot_file(dot_file_path)
                pyg_data = convert_nx_to_pyg_data(nx_graph)

                data_batch.append(pyg_data)
                os.remove(dot_file_path)

                training_index += 1
                count += 1

            except StopIteration:
                print("End of dataset reached.")
                break
            except Exception as e:
                print(f"Error processing LLVM IR: {e}, LLVM IR number: {ir_number}")
                training_index += 1  # Skip to next module
                continue

        if data_batch:
            data_loader = DataLoader(data_batch, batch_size=batch_size)
            loss = train_gnn(model, optimizer, data_loader, loss_fn, device)

            if ir_number % 100 == 0:
                print(f"Epoch {ir_number}, Sample {training_index}, Loss: {loss}")
                # Save the model after each epoch
                torch.save(model.state_dict(), f'./model/model_checkpoint_epoch_{ir_number}.pth')
                print(f"Model checkpoint for epoch {ir_number} saved.")

    # Save the final model
    torch.save(model.state_dict(), './model/final_model.pth')
    print("Final model saved.")

# Run the main function
if __name__ == '__main__':
    main()
