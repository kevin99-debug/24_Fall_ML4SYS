import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
import logging
import sys

def setup_logging():
    """
    Sets up logging to display information and error messages.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_embedding_vector(embedding_str):
    """
    Parses a string of comma-separated floats into a numpy array.
    
    :param embedding_str: String representation of the embedding vector.
    :return: Numpy array of floats.
    """
    try:
        # Remove any leading/trailing whitespace and convert to numpy array
        return np.fromstring(embedding_str.strip('"'), sep=',')
    except Exception as e:
        logging.error(f"Error parsing embedding vector: {e}")
        return np.array([])

def load_embeddings(file_path, has_parsed_embedding=False):
    """
    Loads embeddings from a CSV file and parses the embedding vectors.
    
    :param file_path: Path to the CSV file.
    :param has_parsed_embedding: Boolean indicating if 'parsed_embedding' column exists.
    :return: DataFrame with 'ir_id' and 'parsed_embedding' as numpy arrays.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} records from '{file_path}'.")
    except Exception as e:
        logging.error(f"Failed to load '{file_path}': {e}")
        sys.exit(1)
    
    # Parse the embedding vectors
    if has_parsed_embedding and 'parsed_embedding' in df.columns:
        df['parsed_embedding'] = df['parsed_embedding'].apply(parse_embedding_vector)
    else:
        df['parsed_embedding'] = df['embedding_vector'].apply(parse_embedding_vector)
    
    # Drop any rows with invalid embeddings
    initial_len = len(df)
    df = df[df['parsed_embedding'].apply(lambda x: x.size > 0)]
    if len(df) < initial_len:
        logging.warning(f"Dropped {initial_len - len(df)} records due to parsing errors.")
    
    return df[['ir_id', 'parsed_embedding']]

def assign_clusters(embedding_df, cluster_df):
    """
    Assigns each embedding to the nearest cluster.
    
    :param embedding_df: DataFrame with 'ir_id' and 'parsed_embedding'.
    :param cluster_df: DataFrame with 'cluster_ir_id' and 'parsed_embedding'.
    :return: DataFrame with 'ir_id' and 'cluster_ir_id'.
    """
    # Extract numpy arrays
    embeddings = np.vstack(embedding_df['parsed_embedding'].values)
    cluster_embeddings = np.vstack(cluster_df['parsed_embedding'].values)
    cluster_ids = cluster_df['ir_id'].values  # Assuming 'ir_id' in nearest_vectors.csv is the cluster ID
    
    logging.info(f"Embeddings shape: {embeddings.shape}")
    logging.info(f"Cluster embeddings shape: {cluster_embeddings.shape}")
    
    # Initialize NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(cluster_embeddings)
    logging.info("Fitted NearestNeighbors model.")
    
    # Find the nearest cluster for each embedding
    distances, indices = nbrs.kneighbors(embeddings)
    logging.info("Computed nearest clusters for all embeddings.")
    
    # Assign cluster_ir_id based on nearest cluster
    assigned_cluster_ids = cluster_ids[indices.flatten()]
    
    # Create assignment DataFrame
    assignment_df = pd.DataFrame({
        'ir_id': embedding_df['ir_id'],
        'cluster_ir_id': assigned_cluster_ids
    })
    
    return assignment_df

def main():
    # Set up logging
    setup_logging()
    
    # Load embedding vectors
    logging.info("Loading embedding vectors from embedding.csv...")
    embedding_df = load_embeddings('embedding.csv', has_parsed_embedding=False)
    
    # Load cluster centers
    logging.info("Loading cluster centers from nearest_vectors.csv...")
    cluster_df = load_embeddings('nearest_vectors.csv', has_parsed_embedding=False)
    
    # Assign clusters
    logging.info("Assigning clusters to embeddings...")
    assignment_df = assign_clusters(embedding_df, cluster_df)
    
    # Save the assignments to CSV
    try:
        output_csv = 'nearest_embedding_means.csv'
        assignment_df.to_csv(output_csv, index=False)
        logging.info(f"Cluster assignments saved to '{output_csv}'.")
        count_series = assignment_df.groupby('cluster_ir_id')['ir_id'].count()
        count_df = count_series.reset_index()
        count_df.columns = ['cluster_ir_id', 'ir_count']
        # Sort the clusters by descending order of IR count
        count_df = count_df.sort_values(by='ir_count', ascending=False).reset_index(drop=True)
        print(count_df)
    except Exception as e:
        logging.error(f"Failed to save output CSV '{output_csv}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
