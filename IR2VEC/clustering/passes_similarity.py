import os
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
import logging

def setup_logging(log_file):
    """
    Set up logging to console and a specified log file.

    :param log_file: Path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_pass_logs(logs_dir):
    """
    Load all improving_passes.log files and extract passes.

    :param logs_dir: Directory containing the improving_passes logs.
    :return: Dictionary mapping ir_id to set of passes.
    """
    ir_passes = {}
    for filename in os.listdir(logs_dir):
        if filename.endswith("_improving_passes.log") and filename.startswith("ir_"):
            ir_id_part = filename[len("ir_"):-len("_improving_passes.log")]
            try:
                ir_id = int(ir_id_part)
            except ValueError:
                logging.warning(f"Filename {filename} does not contain a valid ir_id. Skipping.")
                continue
            
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    passes = set(line.strip() for line in f if line.strip())
                ir_passes[ir_id] = passes
                logging.info(f"Loaded {len(passes)} passes for ir_id {ir_id}.")
            except Exception as e:
                logging.error(f"Failed to read {file_path}: {e}")
    
    logging.info(f"Total IRs loaded: {len(ir_passes)}")
    return ir_passes

def build_pass_matrix(ir_passes):
    """
    Build a binary matrix indicating presence of passes for each ir_id.

    :param ir_passes: Dictionary mapping ir_id to set of passes.
    :return: Tuple of (DataFrame with binary indicators, list of ir_ids)
    """
    ir_ids = sorted(ir_passes.keys())
    pass_lists = [ir_passes[ir_id] for ir_id in ir_ids]
    
    mlb = MultiLabelBinarizer()
    pass_matrix = mlb.fit_transform(pass_lists)
    pass_names = mlb.classes_
    
    pass_df = pd.DataFrame(pass_matrix, index=ir_ids, columns=pass_names)
    logging.info(f"Built pass matrix with shape {pass_df.shape}")
    return pass_df, ir_ids

def compute_similarity_matrix(pass_df):
    """
    Compute pairwise similarity matrix between IRs based on passes.

    :param pass_df: DataFrame with binary indicators for passes.
    :param metric: Similarity metric to use ('jaccard' or 'cosine').
    :return: DataFrame containing similarity scores.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(pass_df)
    
    similarity_df = pd.DataFrame(similarity_matrix, index=pass_df.index, columns=pass_df.index)
    logging.info(f"Computed cosine similarity matrix with shape {similarity_df.shape}")
    return similarity_df

def save_similarity_matrix(similarity_df, output_path):
    """
    Save the similarity matrix to a CSV file.

    :param similarity_df: DataFrame containing similarity scores.
    :param output_path: Path to save the CSV file.
    """
    try:
        similarity_df.to_csv(output_path)
        logging.info(f"Similarity matrix saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save similarity matrix to {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compare and compute similarity of improving passes logs.")
    parser.add_argument(
        '-l', '--logs_dir',
        type=str,
        default='./ir/improving_passes',
        help='Directory containing improving_passes.log files (default: ./ir/improving_passes)'
    )
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        default='./pass_similarity_matrix.csv',
        help='Path to save the similarity matrix CSV (default: ./pass_similarity_matrix.csv)'
    )

    args = parser.parse_args()
    
    # Set up logging
    logging.info("Starting the pass similarity comparison process.")
    entries = os.listdir(args.logs_dir)
    # Filter out only directories
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(args.logs_dir, entry))]
    rows = []
    for folder in subfolders:
        current_ir_id = folder.split('_')[1]
        # Load the pass logs
        ir_passes = load_pass_logs(os.path.join(args.logs_dir, folder))
        
        # Build the pass matrix
        pass_df, ir_ids = build_pass_matrix(ir_passes)
        
        # Compute similarity matrix
        similarity_df = compute_similarity_matrix(pass_df)
        try:
            average_similarities = similarity_df.mean(axis=0)
            average_similarities_df = average_similarities.reset_index()
            average_similarities_df.columns = ['ir_id', 'average_similarity']
            average_similarity = average_similarities_df.loc[
                average_similarities_df['ir_id'] == int(current_ir_id), 
                'average_similarity'
            ].values
            rows.append({ 
                'ir_id': current_ir_id, 
                'average_similarity': average_similarity[0]
            })
        except Exception as e:
            logging.error(f"Failed to compute or save average similarities: {e}")
    
    similarity_avg_df = pd.DataFrame(rows)

    logging.info("Pass similarity comparison process completed successfully.")
    similarity_avg_df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
