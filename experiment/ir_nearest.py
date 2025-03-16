import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import subprocess
import os
import csv
from datasets import load_dataset
import argparse

def load_embeddings(csv_path):
    """
    Load embeddings from a CSV file and parse the embedding vectors.
    
    :param csv_path: Path to the embeddings CSV file.
    :return: DataFrame with parsed embeddings.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded embeddings from {csv_path}: {df.shape[0]} records.")
    except Exception as e:
        print(f"Failed to load embeddings from {csv_path}: {e}")
        raise e

    # Parse the 'embedding_vector' column into numerical arrays
    try:
        df['parsed_embedding'] = df['embedding_vector'].apply(lambda x: np.fromstring(x, sep=','))
        print(f"Parsed embedding vectors from column 'embedding_vector'.")
    except Exception as e:
        print(f"Failed to parse embedding vectors in {csv_path}: {e}")
        raise e

    return df

def find_nearest_embeddings(centroids, embeddings, n_neighbors=5):
    """
    For each centroid, find the indices of the nearest 'n_neighbors' embeddings.
    
    :param centroids: Numpy array of centroid vectors.
    :param embeddings: Numpy array of all embedding vectors.
    :param n_neighbors: Number of nearest neighbors to find per centroid.
    :return: Tuple of (indices of nearest embeddings, distances)
    """
    print(f"Finding the {n_neighbors} nearest embeddings for each centroid.")
    # Initialize NearestNeighbors with the desired number of neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embeddings)
    distances, indices = nbrs.kneighbors(centroids)
    print(f"Found nearest embeddings for {centroids.shape[0]} centroids.")
    return indices, distances

def save_ir_file(ir_id, ir_file_path, module_content, graphs_dir):
    """
    Save the IR file corresponding to the given ir_id and generate its call graph.
    
    :param ir_id: The ID of the IR to save.
    :param ir_file_path: Path where the IR file will be saved.
    :param module_content: The content of the module from the dataset.
    :param graphs_dir: Directory to save the call graph DOT files.
    :return: True if successful, False otherwise.
    """
    try:
        # Generate LLVM IR using llvm-dis
        dis_command = ['llvm-dis', '-']
        process = subprocess.Popen(
            dis_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        ir_output, ir_errors = process.communicate(input=module_content)
        if process.returncode != 0:
            print(f"llvm-dis failed for ir_id {ir_id}: {ir_errors.decode('utf-8')}")
            return False

        # Save the IR to the specified path
        with open(ir_file_path, "w") as temp_file:
            temp_file.write(ir_output.decode('utf-8'))
        print(f"Saved IR file to '{ir_file_path}'.")

        # Apply 'opt' to generate the call graph DOT file
        # opt_command = [
        #     'opt',
        #     '-passes=dot-callgraph',
        #     '-disable-output',
        #     ir_file_path
        # ]

        # subprocess.run(opt_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # print(f"Generated call graph for '{ir_file_path}'.")

        # # The generated DOT file is named as ir_file_path + '.callgraph.dot'
        # generated_dot_file = f"{ir_file_path}.callgraph.dot"
        # graph_file_name = f"graph_{ir_id}.dot"
        # new_dot_file_path = os.path.join(graphs_dir, graph_file_name)

        # if os.path.exists(generated_dot_file):
        #     os.rename(generated_dot_file, new_dot_file_path)
        #     print(f"Call graph saved to '{new_dot_file_path}'.")
        # else:
        #     print(f"Expected DOT file '{generated_dot_file}' not found.")
        #     return False

        return True

    except Exception as e:
        print(f"Failed to save IR file for ir_id {ir_id}: {e}")
        return False

def build_ir_mapping(nearest_indices, merged_df):
    """
    Build a mapping from ir_id to list of centroid_ir_ids that require it.
    
    :param nearest_indices: Numpy array of nearest indices (n_centroids x n_neighbors).
    :param merged_df: DataFrame containing the embeddings and metadata.
    :return: Dictionary mapping ir_id to set of centroid_ir_ids.
    """
    ir_to_centroids = {}
    centroid_to_neighbors = {}

    for centroid_idx, neighbors in enumerate(nearest_indices):
        centroid_ir_id = merged_df.iloc[neighbors[0]]['ir_id']
        try:
            centroid_ir_id = int(centroid_ir_id)
        except ValueError:
            print(f"Invalid centroid_ir_id '{centroid_ir_id}' for centroid index {centroid_idx}. Skipping.")
            continue

        centroid_to_neighbors[centroid_ir_id] = []

        for neighbor_idx in neighbors:
            neighbor_ir_id = merged_df.iloc[neighbor_idx]['ir_id']
            try:
                neighbor_ir_id = int(neighbor_ir_id)
            except ValueError:
                print(f"Invalid neighbor_ir_id '{neighbor_ir_id}' for centroid {centroid_ir_id}. Skipping.")
                continue

            centroid_to_neighbors[centroid_ir_id].append(neighbor_ir_id)

            if neighbor_ir_id not in ir_to_centroids:
                ir_to_centroids[neighbor_ir_id] = set()
            ir_to_centroids[neighbor_ir_id].add(centroid_ir_id)

    print(f"Built IR to centroids mapping with {len(ir_to_centroids)} unique IR IDs.")
    return ir_to_centroids, centroid_to_neighbors

def process_dataset(ir_to_centroids, dataset, ir_directory, graphs_dir):
    """
    Iterate through the streaming dataset and save the required IR files.
    
    :param ir_to_centroids: Dictionary mapping ir_id to set of centroid_ir_ids.
    :param dataset: Streaming dataset iterator.
    :param ir_directory: Directory to save IR files.
    :param graphs_dir: Directory to save call graph DOT files.
    """
    try:
        os.makedirs(ir_directory, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)
        print(f"IR files will be saved to '{ir_directory}'.")
        print(f"Call graph DOT files will be saved to '{graphs_dir}'.")

        for ir_id, module in enumerate(dataset):
            if ir_id in ir_to_centroids:
                centroid_ir_ids = ir_to_centroids[ir_id]
                module_content = module['content']

                for centroid_ir_id in centroid_ir_ids:
                    # Directory for each centroid
                    centroid_dir = os.path.join(ir_directory, f"ir_{centroid_ir_id}")
                    os.makedirs(centroid_dir, exist_ok=True)

                    # IR file path
                    ir_file_name = f"ir_{ir_id}.ll"
                    ir_file_path = os.path.join(centroid_dir, ir_file_name)

                    # Save the IR file and generate call graph
                    success = save_ir_file(ir_id, ir_file_path, module_content, graphs_dir)
                    if not success:
                        print(f"Failed to process IR file for ir_id {ir_id} under centroid_ir_id {centroid_ir_id}.")
            if ir_id > 20000:
                break

        print("Completed processing the streaming dataset.")

    except Exception as e:
        print(f"Error while processing the dataset: {e}")
        raise e

def save_cluster_centers(cluster_centers, output_csv):
    """
    Save the K-Means cluster centers to a CSV file with 'embedding_vector' as comma-separated strings.
    
    :param cluster_centers: Numpy array of cluster centers.
    :param output_csv: Path to save the cluster centers CSV.
    """
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['embedding_vector'])
            for cluster_center in cluster_centers:
                embedding_str = ','.join([f"{x:.6f}" for x in cluster_center])
                writer.writerow([embedding_str])
        print(f"Cluster centers saved to {output_csv}")
    except Exception as e:
        print(f"Failed to save cluster centers to {output_csv}: {e}")
        raise e

def parse_arguments():
    """
    Parses command-line arguments.
    
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Nearest Embedding Extraction and IR Saving Script")
    parser.add_argument(
        '-e', '--embedding_csv',
        type=str,
        default='./embedding.csv',
        help='Path to the embeddings CSV file (default: ./embedding.csv)'
    )
    parser.add_argument(
        '-c', '--centroids_csv',
        type=str,
        default='./kmeans_cluster_centers.csv',
        help='Path to the cluster centers CSV file (default: ./kmeans_cluster_centers.csv)'
    )
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        default='./nearest_vectors.csv',
        help='Path to save the nearest embeddings CSV (default: ./nearest_vectors.csv)'
    )
    parser.add_argument(
        '-i', '--ir_directory',
        type=str,
        default='./ir',
        help='Directory to save IR files (default: ./ir)'
    )
    parser.add_argument(
        '-g', '--graphs_directory',
        type=str,
        default='./graphs',
        help='Directory to save call graph DOT files (default: ./graphs)'
    )
    parser.add_argument(
        '-d', '--dataset_name',
        type=str,
        default='llvm-ml/ComPile',
        help='Dataset name from Hugging Face datasets (default: llvm-ml/ComPile)'
    )
    return parser.parse_args()

def main():

    # Parse command-line arguments
    args = parse_arguments()

    print("Starting the nearest embedding extraction process.")

    # Define file paths
    combined_embeddings = args.embedding_csv
    centroids_csv = args.centroids_csv
    output_csv = args.output_csv
    ir_directory = args.ir_directory
    graphs_directory = args.graphs_directory
    dataset_name = args.dataset_name

    # Load embeddings
    combined_df = load_embeddings(combined_embeddings)

    # Load centroids
    centroids_df = load_embeddings(centroids_csv)

    # Extract numerical arrays from 'parsed_embedding'
    embeddings = np.vstack(combined_df['parsed_embedding'].values)
    centroids = np.vstack(centroids_df['parsed_embedding'].values)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Centroids shape: {centroids.shape}")

    # Find nearest embeddings (top 5 per centroid)
    nearest_indices, distances = find_nearest_embeddings(centroids, embeddings, n_neighbors=31)

    # Build mapping from ir_id to centroid_ir_ids
    ir_to_centroids, centroid_to_neighbors = build_ir_mapping(nearest_indices, combined_df)

    # Load the streaming dataset
    print(f"Loading the '{dataset_name}' dataset in streaming mode.")
    try:
        dataset = load_dataset(dataset_name, split='train', streaming=True)
        print("Streaming dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}': {e}")
        return

    # Process the dataset and save required IR files
    process_dataset(ir_to_centroids, dataset, ir_directory, graphs_directory)

    print("Nearest embedding extraction and IR saving process completed successfully.")

if __name__ == "__main__":
    main()
