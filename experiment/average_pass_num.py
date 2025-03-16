import os

def calculate_average_lines_per_folder(base_dir):
    """
    Calculate the average number of lines in log files for each 'ir_{id}' folder.

    :param base_dir: Base directory containing 'ir_{id}' folders.
    :return: Dictionary mapping folder names to the average number of lines per log file.
    """
    averages = {}  # To store the average number of lines for each folder

    # Iterate through all 'ir_{id}' folders
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("ir_"):
            total_lines = 0
            log_file_count = 0

            # Iterate through all files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith("_improving_passes.log"):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        # Count lines in the log file
                        with open(file_path, 'r') as f:
                            line_count = sum(1 for _ in f)
                            total_lines += line_count
                            log_file_count += 1
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
            
            # Calculate the average if there are log files
            if log_file_count > 0:
                averages[folder] = total_lines / log_file_count
                print(f"Folder: {folder}, Total Logs: {log_file_count}, Average Lines: {averages[folder]:.2f}")
            else:
                averages[folder] = 0
                print(f"Folder: {folder} has no log files.")
    
    return averages

def main():
    base_dir = "./ir/improving_passes"  # Replace with your directory path
    averages = calculate_average_lines_per_folder(base_dir)

    print("\nAverage number of lines per log file for each folder:")
    for folder, avg_lines in averages.items():
        print(f"{folder}: {avg_lines:.2f} lines")

if __name__ == "__main__":
    main()