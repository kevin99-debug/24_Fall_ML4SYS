import subprocess
import shutil
import sys
import os
import shutil
import re
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def check_tool_availability(tool_name):
    """
    Checks if a tool is available in the system's PATH.

    :param tool_name: Name of the tool to check.
    :return: Path to the tool if found, else exits the script.
    """
    tool_path = shutil.which(tool_name)
    if tool_path is None:
        print(f"Required tool '{tool_name}' is not found in PATH.")
        sys.exit(1)
    # print(f"Tool '{tool_name}' found at '{tool_path}'.")
    return tool_path

def apply_O3_optimization(input_ll, optimized_ll, passes_log):
    """
    Applies -O3 optimization to the input .ll file and logs the optimization passes.

    :param input_ll: Path to the input .ll file.
    :param optimized_ll: Path to the output optimized .ll file.
    :param passes_log: Path to the log file for optimization passes.
    :return: True if optimization succeeds, False otherwise.
    """
    print(f"Applying -O3 optimization to '{input_ll}'...")
    opt_path = check_tool_availability('opt')

    # Command to apply -O3, enable pass statistics, and capture optimization passes
    command = [
        'opt',
        '-O3',
        '-debug-pass-manager',
        '-stats',
        input_ll,
        '-o',
        optimized_ll
    ]

    try:
        # Run the opt command and capture the output
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )

        # Write the passes and statistics to the passes_log
        with open(passes_log, 'w') as log_fh:
            log_fh.write(result.stdout)

        print(f"Optimization complete. Passes and statistics logged to '{passes_log}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error during optimization of '{input_ll}': {e.output}")
        return False
    return True

def extract_improving_passes(passes_log, improving_passes_log):
    """
    Extracts passes that made improvements from the passes log and saves them to another log file.

    :param passes_log: Path to the log file containing pass outputs and statistics.
    :param improving_passes_log: Path to the log file to save passes that made improvements.
    :return: True if extraction succeeds, False otherwise.
    """
    print(f"Extracting improving passes from '{passes_log}'...")

    improving_passes = set()
    function_instruction_counts = {}  # Maps function names to their current instruction counts

    # Pattern to match lines like:
    # Running pass: SimplifyCFGPass on main (1307 instructions)
    pass_pattern = re.compile(r'^Running pass: (\w+Pass) on (\w+) \((\d+) instructions\)')

    # Open the passes_log and parse line by line
    try:
        with open(passes_log, 'r') as log_fh:
            for line in log_fh:
                line = line.strip()
                match = pass_pattern.match(line)
                if match:
                    pass_name = match.group(1)
                    function_name = match.group(2)
                    instruction_count = int(match.group(3))

                    # Check if we have a previous instruction count for this function
                    if function_name in function_instruction_counts:
                        previous_count = function_instruction_counts[function_name]
                        if instruction_count < previous_count:
                            # Pass made improvements
                            improving_passes.add(pass_name)
                            print(f"Pass '{pass_name}' improved function '{function_name}': {previous_count} -> {instruction_count} instructions.")
                    # Update the current instruction count for the function
                    function_instruction_counts[function_name] = instruction_count
    except FileNotFoundError:
        print(f"Passes log file '{passes_log}' not found.")
        return False
    except Exception as e:
        print(f"Error while extracting improving passes: {e}")
        return False

    if improving_passes:
        # Write the improving passes to the log file
        try:
            with open(improving_passes_log, 'w') as imp_log_fh:
                for imp_pass in sorted(improving_passes):
                    imp_log_fh.write(f"{imp_pass}\n")

            # Also, print the improving passes
            print("Passes that made improvements:")
            for imp_pass in sorted(improving_passes):
                print(imp_pass)

            print(f"Improving passes logged to '{improving_passes_log}'.")
        except Exception as e:
            print(f"Failed to write improving passes to '{improving_passes_log}': {e}")
            return False
    else:
        print("No passes made any modifications.")
        with open(improving_passes_log, 'w') as imp_log_fh:
                imp_log_fh.write(f"NoPasses")

    return True

def compile_to_assembly(input_ll, assembly_s):
    """
    Compiles the given .ll file to assembly using llc.

    :param input_ll: Path to the .ll file.
    :param assembly_s: Path to the output assembly (.s) file.
    :return: True if compilation succeeds, False otherwise.
    """
    print(f"Compiling LLVM IR '{input_ll}' to assembly '{assembly_s}'...")
    llc_path = check_tool_availability('llc')

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
        print(f"Assembly generated at '{assembly_s}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation to assembly for '{input_ll}': {e.stderr}")
        return False
    return True

def analyze_performance(assembly_s, mca_log):
    """
    Analyzes the assembly using llvm-mca and extracts specific performance metrics.

    :param assembly_s: Path to the assembly (.s) file.
    :param mca_log: Path to the log file for llvm-mca output.
    :return: True if analysis succeeds, False otherwise.
    """
    print(f"Analyzing assembly '{assembly_s}' with llvm-mca...")
    llvm_mca_path = check_tool_availability('llvm-mca')

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

        filtered_output = []
        for line in result.stdout.splitlines():
            match = pattern.search(line)
            if match:
                metric_name = match.group(1)
                metric_value = match.group(2)
                filtered_output.append(f"{metric_name} {metric_value}")

        if filtered_output:
            # Write the filtered output to mca_log
            try:
                with open(mca_log, 'w') as log_fh:
                    for line in filtered_output:
                        log_fh.write(line + '\n')

                # Also, print the filtered output
                print("Performance Metrics:")
                for line in filtered_output:
                    print(line)

                print(f"Performance metrics saved to '{mca_log}'.")
            except Exception as e:
                print(f"Failed to write performance metrics to '{mca_log}': {e}")
                return False
        else:
            print("No performance metrics found in llvm-mca output.")

    except subprocess.CalledProcessError as e:
        print(f"Error during llvm-mca analysis for '{assembly_s}': {e.stderr}")
        return False

    return True

def process_ir_file(ir_file_path, output_dir, improving_passes_dir):
    """
    Processes a single LLVM IR (.ll) file through the optimization and analysis pipeline.

    :param ir_file_path: Path to the input .ll file.
    :param output_dir: Directory where output files will be saved.
    :param improving_passes_dir: Directory where improving_passes.log files will be saved.
    """
    try:
        # Extract the base name without extension
        base_name = os.path.splitext(os.path.basename(ir_file_path))[0]
        print(f"\n[File: {base_name}] Processing IR file: '{ir_file_path}'")

        # Define output file paths dynamically based on the base name
        optimized_ll = os.path.join(output_dir, f"{base_name}_O3.ll")
        passes_log = os.path.join(output_dir, f"{base_name}_optimization_passes.log")
        improving_passes_log = os.path.join(improving_passes_dir, f"{base_name}_improving_passes.log")
        assembly_before_s = os.path.join(output_dir, f"{base_name}_before.s")
        mca_before_log = os.path.join(output_dir, f"{base_name}_llvm_mca_before.log")
        assembly_after_s = os.path.join(output_dir, f"{base_name}_O3.s")
        mca_after_log = os.path.join(output_dir, f"{base_name}_llvm_mca_after.log")

        # Step 1: Compile original .ll to assembly_before_s
        if not compile_to_assembly(ir_file_path, assembly_before_s):
            print(f"[File: {base_name}] Skipping further steps due to compilation failure.")
            return

        # Step 2: Analyze performance of original assembly
        if not analyze_performance(assembly_before_s, mca_before_log):
            print(f"[File: {base_name}] Skipping optimization due to analysis failure.")
            return

        # Step 3: Apply -O3 Optimization and log passes
        if not apply_O3_optimization(ir_file_path, optimized_ll, passes_log):
            print(f"[File: {base_name}] Skipping further steps due to optimization failure.")
            return

        # Step 4: Extract and log improving passes
        if not extract_improving_passes(passes_log, improving_passes_log):
            print(f"[File: {base_name}] No improving passes extracted.")

        # Step 5: Compile optimized .ll to assembly_after_s
        if not compile_to_assembly(optimized_ll, assembly_after_s):
            print(f"[File: {base_name}] Skipping performance analysis of optimized assembly due to compilation failure.")
            return

        # Step 6: Analyze performance of optimized assembly
        if not analyze_performance(assembly_after_s, mca_after_log):
            print(f"[File: {base_name}] Performance analysis of optimized assembly failed.")
            return

        print(f"[File: {base_name}] Completed processing successfully.\n")

    except Exception as e:
        print(f"[File: {base_name}] An unexpected error occurred: {e}")

def parse_arguments():
    """
    Parses command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="LLVM IR Optimization and Performance Analysis Script")
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default='./ir',
        help='Directory containing input .ll files (default: ./ir)'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default='./ir/output',
        help='Directory to store output files (default: ./ir/output)'
    )
    parser.add_argument(
        '-m', '--improving_passes_dir',
        type=str,
        default='./ir/improving_passes',
        help='Directory to store improving_passes.log files (default: ./ir/improving_passes)'
    )
    parser.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='Enable parallel processing of IR files'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    print("LLVM Optimization and Performance Analysis Script Started.")

    entries = os.listdir(args.input_dir)
    # Filter out only directories
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(args.input_dir, entry))]
    print(subfolders)
    for folder in subfolders:
        if 'ir' not in folder:
            continue
        current_ir_id = folder.split('_')[1]

        print(f"LLVM Optimization and Performance Analysis Scripting on IR ID {current_ir_id}")
        # Define the directory containing IR files
        ir_directory = args.input_dir + f'/ir_{current_ir_id}'
        output_directory = args.output_dir + f'/ir_{current_ir_id}'
        improving_passes_directory = args.improving_passes_dir + f'/ir_{current_ir_id}'

        # Create output directories if they don't exist
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(improving_passes_directory, exist_ok=True)

        # Find all .ll files in the ir_directory
        ir_files = glob(os.path.join(ir_directory, '*.ll'))

        if not ir_files:
            print(f"No .ll files found in directory '{ir_directory}'. Exiting script.")
            sys.exit(0)

        print(f"Found {len(ir_files)} .ll files to process in '{ir_directory}'.")

        if args.parallel:
            print(f"Processing IR files in parallel using {args.workers} workers.")

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all tasks
                future_to_ir = {
                    executor.submit(process_ir_file, ir_file, output_directory, improving_passes_directory): ir_file
                    for ir_file in ir_files
                }

                # Process as they complete
                for future in as_completed(future_to_ir):
                    ir_file = future_to_ir[future]
                    try:
                        future.result()
                    except Exception as e:
                        base_name = os.path.splitext(os.path.basename(ir_file))[0]
                        print(f"[File: {base_name}] An error occurred during processing: {e}")
        else:
            # Process files sequentially
            for ir_file in ir_files:
                process_ir_file(ir_file, output_directory, improving_passes_directory)

    print("All IR files have been processed.")

    shutil.rmtree('./ir/output')

if __name__ == "__main__":
    main()
