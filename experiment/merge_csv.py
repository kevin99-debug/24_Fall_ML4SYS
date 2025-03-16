import pandas as pd
import ast
import csv
import os
import logging

def load_and_prepare_csv(graph_embeddings_path, for_loop_embeddings_path):
    """
    Load the graph embeddings and other embeddings CSV files,
    and parse the embedding vectors from strings to lists.
    """
    try:
        graph_df = pd.read_csv(graph_embeddings_path)
        logging.info(f"Loaded Graph Embeddings: {graph_df.shape[0]} records.")
    except Exception as e:
        logging.error(f"Failed to load graph embeddings CSV: {e}")
        raise e

    try:
        other_df = pd.read_csv(for_loop_embeddings_path)
        logging.info(f"Loaded Other Embeddings: {other_df.shape[0]} records.")
    except Exception as e:
        logging.error(f"Failed to load other embeddings CSV: {e}")
        raise e

    try:
        graph_df['embedding_vector'] = graph_df['embedding_vector'].apply(ast.literal_eval)
        other_df['embedding_vector'] = other_df['embedding_vector'].apply(ast.literal_eval)
        logging.info("Parsed embedding vectors successfully.")
    except Exception as e:
        logging.error(f"Failed to parse embedding vectors: {e}")
        raise e

    return graph_df, other_df

def merge_embeddings(graph_df, other_df):
    """
    Merge the two DataFrames on ir_id and ir_index,
    concatenate their embedding vectors, and remove unmatched rows.
    """
    # Rename columns for clarity before merging
    graph_df = graph_df.rename(columns={'ir_index': 'ir_id', 'embedding_vector': 'graph_embedding'})
    # other_df already has 'ir_id'

    # Merge DataFrames on 'ir_id'
    merged_df = pd.merge(other_df, graph_df, on='ir_id', how='inner')
    logging.info(f"Merged DataFrame: {merged_df.shape[0]} records after merging.")

    # Concatenate the two embedding vectors
    def safe_concat(row):
        if isinstance(row['embedding_vector'], list) and isinstance(row['graph_embedding'], list):
            return row['embedding_vector'] + row['graph_embedding']
        else:
            logging.warning(f"Invalid embedding vectors for ir_id {row['ir_id']}.")
            return []

    merged_df['combined_embedding'] = merged_df.apply(safe_concat, axis=1)

    # Drop the original embedding_vector columns
    merged_df = merged_df.drop(['embedding_vector', 'graph_embedding'], axis=1)
    
    expected_total_dim = 800
    combined_lengths = merged_df['combined_embedding'].apply(len)
    combined_length_counts = combined_lengths.value_counts()

    logging.info(f"Combined Embedding Vector Lengths:\n{combined_length_counts}")

    # Identify incorrect lengths
    incorrect_length_df = merged_df[combined_lengths != expected_total_dim]

    if not incorrect_length_df.empty:
        logging.warning(f"Found {incorrect_length_df.shape[0]} rows with incorrect combined embedding lengths.")
        logging.warning(f"These rows will be excluded from the merged embeddings.")
        # Remove these rows
        merged_df = merged_df[combined_lengths == expected_total_dim]
    else:
        logging.info("All combined embedding vectors have the correct length.")

    merged_df = merged_df.rename(columns={'combined_embedding': 'embedding_vector'})
    return merged_df

def save_merged_embeddings(merged_df, output_csv_path):
    """
    Save the merged DataFrame with concatenated embeddings to a CSV file.
    The embedding vector is stored as a single comma-separated string in one column.
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['ir_id', 'embedding_vector'])
            # Write data rows
            for _, row in merged_df.iterrows():
                # Check if embedding_vector is not empty
                if not row['embedding_vector']:
                    logging.warning(f"Skipping ir_id {row['ir_id']} due to empty embedding.")
                    continue
                # Convert the embedding_vector list to a comma-separated string
                embedding_str = ','.join([f"{x:.6f}" for x in row['embedding_vector']])
                writer.writerow([row['ir_id'], embedding_str])

        logging.info(f"Merged embeddings saved to {output_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save merged embeddings CSV: {e}")
        raise e

def main():
    logging.info("Starting the embedding merge process.")
    graph_embeddings = './embedding_vectors_test.csv'
    bert_embeddings = './embedding_vectors_all_test.csv'

    # Validate input file paths
    if not os.path.isfile(graph_embeddings):
        logging.error(f"Graph embeddings file not found: {graph_embeddings}")
        return
    if not os.path.isfile(bert_embeddings):
        logging.error(f"Other embeddings file not found: {bert_embeddings}")
        return

    # Load and prepare CSV files
    try:
        graph_df, other_df = load_and_prepare_csv(graph_embeddings, bert_embeddings)
    except Exception as e:
        logging.error(f"Terminating due to error in loading/parsing CSVs: {e}")
        return

    # Merge embeddings
    merged_df = merge_embeddings(graph_df, other_df)

    # Check if merged_df is empty
    if merged_df.empty:
        logging.warning("No matching ir_id and ir_index found. Exiting without saving.")
        return

    # Save the merged embeddings to CSV
    output_file_dir = './embedding.csv'
    try:
        save_merged_embeddings(merged_df, output_file_dir)
    except Exception as e:
        logging.error(f"Terminating due to error in saving merged embeddings: {e}")
        return

    logging.info("Embedding merge process completed successfully.")

if __name__ == "__main__":
    main()
