import pandas as pd
import numpy as np
import subprocess
import sys
import os
import re
import argparse
import logging
from datasets import load_dataset
import time

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

def compile_to_assembly(input_ll, assembly_s):
    """
    Compiles the given .ll file to assembly using llc.

    :param input_ll: Path to the .ll file.
    :param assembly_s: Path to the output assembly (.s) file.
    :return: True if compilation succeeds, False otherwise.
    """
    logging.info(f"Compiling LLVM IR '{input_ll}' to assembly '{assembly_s}'...")

    command = [
        'llc',
        '-filetype=asm',
        input_ll,
        '-o',
        assembly_s
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logging.info(f"Assembly generated at '{assembly_s}'.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during compilation to assembly for '{input_ll}': {e.stderr}")
        return False
    return True

def analyze_performance(assembly_s, mca_log):
    """
    Analyzes the assembly using llvm-mca and extracts specific performance metrics.

    :param assembly_s: Path to the assembly (.s) file.
    :param mca_log: Path to the log file for llvm-mca output.
    :return: Dictionary of extracted metrics if successful, else None.
    """
    logging.info(f"Analyzing assembly '{assembly_s}' with llvm-mca...")

    # Command to run llvm-mca
    command = [
        'llvm-mca',
        '-iterations=1',
        assembly_s
    ]

    try:
        # Run llvm-mca
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Define the metrics to extract
        metrics = [
            "Iterations:",
            "Instructions:",
            "Total Cycles:",
            "Total uOps:",
            "Dispatch Width:",
            "uOps Per Cycle:",
            "IPC:",
            "Block RThroughput:"
        ]

        # Compile a regex pattern to match the metrics
        pattern = re.compile(r'(' + '|'.join([re.escape(m) for m in metrics]) + r')\s+([\d.]+)')

        extracted_metrics = {}
        for line in result.stdout.splitlines():
            match = pattern.search(line)
            if match:
                metric_name = match.group(1).rstrip(':')  # Remove the colon
                metric_value = match.group(2)
                extracted_metrics[metric_name] = float(metric_value)

        if extracted_metrics:
            # Write the filtered output to mca_log
            try:
                with open(mca_log, 'w') as log_fh:
                    for metric, value in extracted_metrics.items():
                        log_fh.write(f"{metric}: {value}\n")

                logging.info(f"Performance metrics saved to '{mca_log}'.")
            except Exception as e:
                logging.error(f"Failed to write performance metrics to '{mca_log}': {e}")
                return None

            return extracted_metrics
        else:
            logging.warning("No performance metrics found in llvm-mca output.")
            return None

    except subprocess.CalledProcessError as e:
        logging.error(f"Error during llvm-mca analysis for '{assembly_s}': {e.stderr}")
        return None


def load_improving_passes(log_file_path):
    """
    Loads the list of improving passes from a log file.

    :param log_file_path: Path to the improving_passes.log file.
    :return: List of pass names.
    """
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        # Remove any whitespace and empty lines
        passes = [line.strip() for line in lines if line.strip()]
        if not passes:
            logging.warning(f"No passes found in '{log_file_path}'.")
        return passes
    except FileNotFoundError:
        logging.error(f"Improving passes log file '{log_file_path}' not found.")
        return []
    except Exception as e:
        logging.error(f"Error reading '{log_file_path}': {e}")
        return []

def apply_passes(ir_file, optimized_ir_file, pass_list):
    """
    Applies a list of optimization passes to an IR file using llvm-opt.

    :param ir_file: Path to the input .ll file.
    :param optimized_ir_file: Path to save the optimized .ll file.
    :param pass_list: List of pass names to apply.
    :return: True if successful, False otherwise.
    """
    
    # Construct the opt command with the list of passes
    passes = ""
    if "O3" in pass_list:
        passes = '-O3'
    else:
        passes = '-passes=' + ','.join(pass_list)
    command = ['opt', passes]
    command.extend(['-S', ir_file, '-o', optimized_ir_file])
    
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logging.info(f"Applied passes to '{ir_file}' and saved optimized IR to '{optimized_ir_file}'.")
        return time.time() - start_time
    except subprocess.CalledProcessError as e:
        logging.error(f"Error applying passes to '{ir_file}': {e.stderr}")
        return 0

def process_assignment(row, optimize_passes_dir, output_dir, ir_file_path):
    """
    Processes a single IR assignment: applies passes and counts instructions.

    :param row: A row from the cluster_assignments DataFrame.
    :param dataset_dir: Directory containing the IR .ll files.
    :param optimize_passes_dir: Directory containing optimizing passes logs.
    :param output_dir: Directory to save optimized IR files.
    :return: Dictionary with ir_id, cluster_ir_id, instruction_count_before, instruction_count_after.
    """
    ir_id = row['ir_id']
    cluster_ir_id = row['cluster_ir_id']
    
    if not os.path.isfile(ir_file_path):
        logging.error(f"IR file '{ir_file_path}' not found.")
        return {
            'ir_id': ir_id,
            'cluster_ir_id': cluster_ir_id,
            'instruction_count_before': 'FileNotFound',
            'instruction_count_after': 'FileNotFound',
            'uops_per_cycle_before': 'FileNotFound',
            'uops_per_cycle_after': 'FileNotFound',
            'IPC_before': 'FileNotFound',
            'IPC_after': 'FileNotFound',
            'block_rthroughput_before': 'FileNotFound',
            'block_rthroughput_after': 'FileNotFound',
            'vlos_time': 0,
            'o3_time': 0
        }
    
    passes_log_path = os.path.join(optimize_passes_dir, f"ir_{cluster_ir_id}_improving_passes.log")
    pass_list = load_improving_passes(passes_log_path)
    
    # Count instructions before optimization
    assembly_before_s = os.path.join(output_dir, "temp_before.s")
    mca_before_log = os.path.join(output_dir, "temp_before.log")
    count_before = 0
    uops_before = 0
    ipc_before = 0
    block_before = 0

    if not compile_to_assembly(ir_file_path, assembly_before_s):
        logging.error(f"Failed to compile '{ir_file_path}' to assembly.")
        count_before = 'CompileFailed'
    else:
        metrics_before = analyze_performance(assembly_before_s, mca_before_log)
        count_before = metrics_before["Instructions"]
        uops_before = metrics_before["uOps Per Cycle"]
        ipc_before = metrics_before["IPC"]
        block_before = metrics_before["Block RThroughput"]

    if not pass_list:
        # logging.warning(f"No improving passes for cluster_ir_id '{cluster_ir_id}'. Skipping IR '{ir_id}'.")
        return {
            'ir_id': ir_id,
            'cluster_ir_id': cluster_ir_id,
            'instruction_count_before': count_before,
            'instruction_count_after': count_before,
            'instruction_count_o3': count_before,
            'uops_per_cycle_before': uops_before,
            'uops_per_cycle_after': uops_before,
            'uops_per_cycle_o3': uops_before,
            'IPC_before': ipc_before,
            'IPC_after': ipc_before,
            'IPC_o3': ipc_before,
            'block_rthroughput_before': block_before,
            'block_rthroughput_after': block_before,
            'block_rthroughput_o3': block_before,
            'vlos_time': 0,
            'o3_time': 0
        }

    # Define optimized IR file path
    optimized_ir_file = os.path.join(output_dir, f"temp_optimized.ll")
    count_after = 0
    uops_after = 0
    ipc_after = 0
    block_after = 0
    vlos_exec_time = 0
    if not pass_list or 'NoPasses' in pass_list:
        # logging.warning(f"No improving passes for cluster_ir_id '{cluster_ir_id}'. Skipping IR '{ir_id}'.")
        count_after = count_before
        uops_after = uops_before
        ipc_after = ipc_before
        block_after = block_before
    else:
        vlos_exec_time = apply_passes(ir_file_path, optimized_ir_file, pass_list)
        if not vlos_exec_time:
            logging.error(f"Failed to apply passes to IR '{ir_id}'.")
            count_after = 'OptFailed'
        else:
            # Step 3: Count instructions after optimization
            assembly_after_s = os.path.join(output_dir, f"temp_after.s")
            mca_after_log = os.path.join(output_dir, f"temp_after.log")
            
            if not compile_to_assembly(optimized_ir_file, assembly_after_s):
                logging.error(f"Failed to compile optimized IR '{optimized_ir_file}' to assembly.")
                count_after = 'CompileFailed'
            else:
                metrics_after = analyze_performance(assembly_after_s, mca_after_log)
                count_after = metrics_after["Instructions"]
                uops_after = metrics_after["uOps Per Cycle"]
                ipc_after = metrics_after["IPC"]
                block_after = metrics_after["Block RThroughput"]
    
    # Define optimized IR file path
    optimized_ir_file_o3 = os.path.join(output_dir, f"temp_optimized_o3.ll")
    count_o3 = 0
    uops_o3 = 0
    ipc_o3 = 0
    block_o3 = 0
    o3_exec_time = apply_passes(ir_file_path, optimized_ir_file_o3, ["O3"])
    if not o3_exec_time:
        logging.error(f"Failed to apply passes to IR '{ir_id}'.")
        count_o3 = 'OptFailed'
    else:
        # Step 3: Count instructions after optimization
        assembly_s_o3 = os.path.join(output_dir, f"temp_o3.s")
        mca_log_o3 = os.path.join(output_dir, f"temp_o3.log")
        
        if not compile_to_assembly(optimized_ir_file_o3, assembly_s_o3):
            logging.error(f"Failed to compile optimized IR '{optimized_ir_file_o3}' to assembly.")
            count_o3 = 'CompileFailed'
        else:
            metrics_o3 = analyze_performance(assembly_s_o3, mca_log_o3)
            count_o3 = metrics_o3["Instructions"]
            uops_o3 = metrics_o3["uOps Per Cycle"]
            ipc_o3 = metrics_o3["IPC"]
            block_o3 = metrics_o3["Block RThroughput"]
            
    return {
        'ir_id': ir_id,
        'cluster_ir_id': cluster_ir_id,
        'instruction_count_before': count_before,
        'instruction_count_after': count_after,
        'instruction_count_o3': count_o3,
        'uops_per_cycle_before': uops_before,
        'uops_per_cycle_after':  uops_after,
        'uops_per_cycle_o3':  uops_o3,
        'IPC_before': ipc_before,
        'IPC_after': ipc_after,
        'IPC_o3': ipc_o3,
        'block_rthroughput_before': block_before,
        'block_rthroughput_after': block_after,
        'block_rthroughput_o3': block_o3,
        'vlos_time': vlos_exec_time,
        'o3_time': o3_exec_time
    }

def main():
    # Set up logging
    setup_logging()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Assign optimization passes to IR files and count instructions.")
    parser.add_argument(
        '--cluster_assignments_csv',
        type=str,
        default='./nearest_embedding_means.csv',
        help='Path to the cluster_assignments.csv file.'
    )
    parser.add_argument(
        '--optimize_passes_dir',
        type=str,
        default='./optimize_passes',
        help='Directory containing the improving_passes.log files.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./ir_output',
        help='Directory to save optimized IR files.'
    )
    parser.add_argument(
        '--result_csv',
        type=str,
        default='./result.csv',
        help='Path to save the result CSV file.'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load cluster assignments
    cluster_df = pd.read_csv(args.cluster_assignments_csv)
    
    results = []

    # Sequential processing
    logging.info("Processing IR files sequentially.")

    ds = load_dataset('llvm-ml/ComPile', split='train', streaming=True)

    df_ir_index = 0
    current_ir_id = cluster_df.iloc[df_ir_index]['ir_id']
    last_ir_id = cluster_df.iloc[-1]['ir_id']
    
    for i, module in enumerate(ds):
        if i > last_ir_id or df_ir_index > 1000:
            break
        current_ir_id = cluster_df.iloc[df_ir_index]['ir_id']
        if i < current_ir_id:
            continue        
        try:
            bitcode_module = module['content']  # 이미 bytes 형식임
            dis_command = ['llvm-dis', '-']
            with subprocess.Popen(
                dis_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            ) as process:
                ir_output, _ = process.communicate(input=bitcode_module)

            with open("./ir_temp.ll", "w") as temp_file:
                temp_file.write(ir_output.decode('utf-8'))

            # 진행 상황 확인용 출력
            print(f"Saved ir {i}, current ir id {current_ir_id}, current ir index {df_ir_index}")
            result = process_assignment(cluster_df.iloc[df_ir_index], args.optimize_passes_dir, args.output_dir, "./ir_temp.ll")
            if (result['instruction_count_before'] not in ['FileNotFound', 'CompileFailed', 'OptFailed']) \
                and (result['instruction_count_after'] not in ['FileNotFound', 'CompileFailed', 'OptFailed']) \
                and (result['instruction_count_o3'] not in ['FileNotFound', 'CompileFailed', 'OptFailed']):
                results.append(result)

            df_ir_index += 1
        except Exception as e:
            print(f"Error processing module {i}: {e}")
            continue

    # Create result DataFrame
    result_df = pd.DataFrame(results)
    
    # Save to CSV
    try:
        result_df.to_csv(args.result_csv, index=False)
        logging.info(f"Results saved to '{args.result_csv}'.")
    except Exception as e:
        logging.error(f"Failed to save results to '{args.result_csv}': {e}")
        sys.exit(1)
    
    logging.info("Processing completed successfully.")

if __name__ == "__main__":
    main()
