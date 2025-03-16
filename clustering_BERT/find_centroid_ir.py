import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
import subprocess
import os
import logging
import csv
from ast import literal_eval
from datasets import load_dataset

def load_embeddings(csv_path):
    """
    Load embeddings from a CSV file and parse the embedding vectors.
    """
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded embeddings from {csv_path}: {df.shape[0]} records.")
    except Exception as e:
        logging.error(f"Failed to load embeddings from {csv_path}: {e}")
        raise e

    # Parse the 'embedding_vector' column into numerical arrays
    try:
        df['parsed_embedding'] = df['embedding_vector'].apply(lambda x: np.fromstring(x, sep=','))
        logging.info(f"Parsed embedding vectors from column 'embedding_vector'.")
    except Exception as e:
        logging.error(f"Failed to parse embedding vectors in {csv_path}: {e}")
        raise e

    return df

def find_nearest_embeddings(centroids, embeddings):
    """
    For each centroid, find the index of the nearest embedding.
    Returns a list of indices corresponding to the nearest embeddings.
    """
    # Compute the nearest embeddings for each centroid
    closest_indices, distances = pairwise_distances_argmin_min(centroids, embeddings)
    logging.info(f"Computed nearest embeddings for {centroids.shape[0]} centroids.")
    return closest_indices, distances

def save_nearest_rows_and_ir(merged_df, nearest_indices, output_csv):
    """
    Save the rows corresponding to the nearest embeddings to a new CSV file.
    """
    try:
        nearest_rows = merged_df.iloc[nearest_indices].copy()
        logging.info(f"Selected {nearest_rows.shape[0]} nearest embedding rows.")
        
        # Optional: Reset index if desired
        nearest_rows.reset_index(drop=True, inplace=True)

        # Save to CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        nearest_rows.to_csv(output_csv, index=False)
        logging.info(f"Saved nearest embeddings to {output_csv}")
        dataset = load_dataset('llvm-ml/ComPile', split='train', streaming=True)
        os.makedirs('ir', exist_ok=True)
        os.makedirs('graphs', exist_ok=True)

        for _, nearest_row in nearest_rows.iterrows():
            print(nearest_row['ir_id'])
            ir_index = nearest_row['ir_id']

            for i, module in enumerate(dataset):  # Ensure module is the dataset entry
                if i < ir_index:
                    continue
                # Generate LLVM IR
                dis_command = ['llvm-dis', '-']
                with subprocess.Popen(
                    dis_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                ) as process:
                    ir_output, _ = process.communicate(input=module['content'])

                ir_file_path = f"ir/ir_{ir_index}.ll"
                with open(ir_file_path, "w") as temp_file:
                    temp_file.write(ir_output.decode('utf-8'))

                command = [
                    'opt',
                    '-passes=dot-callgraph',
                    '-disable-output',
                    ir_file_path
                ]

                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                # The generated DOT file is named as ir_file_path + '.callgraph.dot'
                generated_dot_file = ir_file_path + '.callgraph.dot'
                new_dot_file_path = f'graphs/graph_{ir_index}.dot'

                if os.path.exists(generated_dot_file):
                    os.rename(generated_dot_file, new_dot_file_path)
                else:
                    raise FileNotFoundError(f"Expected DOT file {generated_dot_file} not found.")
                break

    except Exception as e:
        logging.error(f"Failed to save nearest embeddings to {output_csv}: {e}")
        raise e

def save_cluster_centers(cluster_centers, output_csv):
    """
    Save the K-Means cluster centers to a CSV file with 'embedding_vector' as comma-separated strings.
    """
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['embedding_vector'])
            for cluster_center in cluster_centers:
                embedding_str = ','.join([f"{x:.6f}" for x in cluster_center])
                writer.writerow([embedding_str])
        logging.info(f"Cluster centers saved to {output_csv}")
    except Exception as e:
        logging.error(f"Failed to save cluster centers to {output_csv}: {e}")
        raise e

def main():
    logging.info("Starting the nearest embedding extraction process.")

    # Define file paths
    combined_embeddings = './embedding.csv'
    centroids_csv = './kmeans_cluster_centers.csv'
    output_csv = './nearest_vectors.csv'

    # Load embeddings
    combined_df = load_embeddings(combined_embeddings)

    # Load centroids
    centroids_df = load_embeddings(centroids_csv)

    # Extract numerical arrays from 'parsed_embedding'
    embeddings = np.vstack(combined_df['parsed_embedding'].values)
    centroids = np.vstack(centroids_df['parsed_embedding'].values)

    logging.info(f"Embeddings shape: {embeddings.shape}")
    logging.info(f"Centroids shape: {centroids.shape}")

    # Find nearest embeddings
    nearest_indices, distances = find_nearest_embeddings(centroids, embeddings)

    # Save the nearest rows to CSV
    save_nearest_rows_and_ir(combined_df, nearest_indices, output_csv)

    logging.info("Nearest embedding extraction process completed successfully.")

if __name__ == "__main__":
    main()
