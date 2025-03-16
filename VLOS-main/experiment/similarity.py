import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# List of all 70 passes
PASS_LIST = [
    "Annotation2MetadataPass", "ForceFunctionAttrsPass", "InferFunctionAttrsPass", "CoroEarlyPass",
    "LowerExpectIntrinsicPass", "SimplifyCFGPass", "SROAPass", "EarlyCSEPass", "CallSiteSplittingPass",
    "OpenMPOptPass", "IPSCCPPass", "CalledValuePropagationPass", "GlobalOptPass", "PromotePass",
    "InstCombinePass", "AlwaysInlinerPass", "ModuleInlinerWrapperPass", "InvalidateAnalysisPass",
    "PostOrderFunctionAttrsPass", "ArgumentPromotionPass", "OpenMPOptCGSCCPass", "SpeculativeExecutionPass",
    "JumpThreadingPass", "CorrelatedValuePropagationPass", "AggressiveInstCombinePass", "LibCallsShrinkWrapPass",
    "TailCallElimPass", "ReassociatePass", "ConstraintEliminationPass", "LoopSimplifyPass", "LCSSAPass",
    "VectorCombinePass", "MergedLoadStoreMotionPass", "GVNPass", "SCCPPass", "BDCEPass", "ADCEPass",
    "MemCpyOptPass", "DSEPass", "MoveAutoInitPass", "CoroElidePass", "CoroSplitPass", "DeadArgumentEliminationPass",
    "CoroCleanupPass", "GlobalDCEPass", "EliminateAvailableExternallyPass", "ReversePostOrderFunctionAttrsPass",
    "RecomputeGlobalsAAPass", "Float2IntPass", "LowerConstantIntrinsicsPass", "ControlHeightReductionPass",
    "LoopDistributePass", "InjectTLIMappings", "LoopVectorizePass", "InferAlignmentPass", "LoopLoadEliminationPass",
    "SLPVectorizerPass", "LoopUnrollPass", "WarnMissedTransformationsPass", "AlignmentFromAssumptionsPass",
    "LoopSinkPass", "InstSimplifyPass", "DivRemPairsPass", "ConstantMergePass", "CGProfilePass",
    "RelLookupTableConverterPass", "AnnotationRemarksPass", "VerifierPass", "BitcodeWriterPass", "NoPasses"
]

# Mapping of pass names to their index in the binary vector
PASS_INDEX = {pass_name: i for i, pass_name in enumerate(PASS_LIST)}

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
                print(f"Filename {filename} does not contain a valid ir_id. Skipping.")
                continue
            
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    passes = set(line.strip() for line in f if line.strip())
                ir_passes[ir_id] = passes
                print(f"Loaded {len(passes)} passes for ir_id {ir_id}.")
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
    
    print(f"Total IRs loaded: {len(ir_passes)}")
    return ir_passes

def build_pass_matrix(ir_passes):
    """
    Build a binary matrix indicating presence of passes for each ir_id.

    :param ir_passes: Dictionary mapping ir_id to set of passes.
    :return: Tuple of (DataFrame with binary indicators, list of ir_ids)
    """
    ir_ids = sorted(ir_passes.keys())
    pass_lists = []

    for ir_id in ir_ids:
        # Initialize the binary vector for the IR with all zeros
        binary_vector = np.zeros(len(PASS_LIST), dtype=int)
        # Mark the presence of each pass by setting the corresponding index to 1
        for pass_name in ir_passes[ir_id]:
            if pass_name in PASS_INDEX:
                binary_vector[PASS_INDEX[pass_name]] = 1
        pass_lists.append(binary_vector)
    
    # Create a DataFrame for the binary pass vectors
    pass_df = pd.DataFrame(pass_lists, index=ir_ids, columns=PASS_LIST)
    print(f"Built pass matrix with shape {pass_df.shape}")
    return pass_df, ir_ids

def compute_cosine_similarity(pass_df):
    """
    Compute pairwise cosine similarity matrix.
    """
    similarity_matrix = cosine_similarity(pass_df)
    similarity_df = pd.DataFrame(similarity_matrix, index=pass_df.index, columns=pass_df.index)
    return similarity_df

def compute_jaccard_similarity(pass_df):
    """
    Compute pairwise Jaccard similarity matrix.
    """
    num_ir = pass_df.shape[0]
    jaccard_matrix = np.zeros((num_ir, num_ir))

    for i in range(num_ir):
        for j in range(num_ir):
            intersection = np.sum(np.logical_and(pass_df.iloc[i], pass_df.iloc[j]))
            union = np.sum(np.logical_or(pass_df.iloc[i], pass_df.iloc[j]))
            jaccard_matrix[i, j] = intersection / union if union != 0 else 0
    
    return pd.DataFrame(jaccard_matrix, index=pass_df.index, columns=pass_df.index)

def compute_hamming_distance(pass_df):
    """
    Compute pairwise Hamming distance matrix.
    """
    num_ir = pass_df.shape[0]
    hamming_matrix = np.zeros((num_ir, num_ir))

    for i in range(num_ir):
        for j in range(num_ir):
            hamming_matrix[i, j] = np.sum(pass_df.iloc[i] != pass_df.iloc[j])
    
    return pd.DataFrame(hamming_matrix, index=pass_df.index, columns=pass_df.index)

def compute_tanimoto_coefficient(pass_df):
    """
    Compute pairwise Tanimoto coefficient matrix.
    """
    num_ir = pass_df.shape[0]
    tanimoto_matrix = np.zeros((num_ir, num_ir))

    for i in range(num_ir):
        for j in range(num_ir):
            intersection = np.sum(np.logical_and(pass_df.iloc[i], pass_df.iloc[j]))
            sum_square = np.sum(pass_df.iloc[i]) + np.sum(pass_df.iloc[j])
            tanimoto_matrix[i, j] = intersection / (sum_square - intersection) if sum_square != intersection else 0
    
    return pd.DataFrame(tanimoto_matrix, index=pass_df.index, columns=pass_df.index)

def save_similarity_matrices(similarity_dfs, output_path):
    """
    Save all similarity matrices to CSV files.
    """
    for name, similarity_df in similarity_dfs.items():
        output_file = f"{output_path}_{name}.csv"
        try:
            similarity_df.to_csv(output_file)
            print(f"Similarity matrix saved to {output_file}")
        except Exception as e:
            print(f"Failed to save {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compare and compute similarity of improving passes logs.")
    parser.add_argument(
        '-l', '--logs_dir',
        type=str,
        default='./ir/improving_passes_ir2vec',
        help='Directory containing improving_passes.log files (default: ./ir/improving_passes)'
    )
    parser.add_argument(
        '-o', '--output_csv',
        type=str,
        default='./pass_similarity_matrix',
        help='Path to save the similarity matrix CSV (default: ./pass_similarity_matrix)'
    )

    args = parser.parse_args()
    
    # Set up logging
    print("Starting the pass similarity comparison process.")
    
    # Load pass logs from subdirectories
    entries = os.listdir(args.logs_dir)
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(args.logs_dir, entry))]
    
    rows = []
    for folder in subfolders:
        current_ir_id = folder.split('_')[1]
        ir_passes = load_pass_logs(os.path.join(args.logs_dir, folder))
        
        # Build the pass matrix
        pass_df, ir_ids = build_pass_matrix(ir_passes)
        
        # Compute similarity matrices
        similarity_dfs = {
            'cosine': compute_cosine_similarity(pass_df),
            'jaccard': compute_jaccard_similarity(pass_df),
            'hamming': compute_hamming_distance(pass_df),
            'tanimoto': compute_tanimoto_coefficient(pass_df)
        }

        # Save all similarity matrices
        save_similarity_matrices(similarity_dfs, args.output_csv)

        # Compute and save average similarities for the current IR
        for name, similarity_df in similarity_dfs.items():
            try:
                average_similarities = similarity_df.mean(axis=0)
                average_similarities_df = average_similarities.reset_index()
                average_similarities_df.columns = ['ir_id', f'average_similarity_{name}']
                average_similarity = average_similarities_df.loc[
                    average_similarities_df['ir_id'] == int(current_ir_id), 
                    f'average_similarity_{name}'
                ].values
                rows.append({ 
                    'ir_id': current_ir_id, 
                    f'average_similarity_{name}': average_similarity[0]
                })
            except Exception as e:
                print(f"Failed to compute or save average similarities for {name}: {e}")
    
    # Save final results
    similarity_avg_df = pd.DataFrame(rows)
    similarity_avg_df.to_csv(args.output_csv + '_average.csv', index=False)

    print("Pass similarity comparison process completed successfully.")

if __name__ == "__main__":
    main()