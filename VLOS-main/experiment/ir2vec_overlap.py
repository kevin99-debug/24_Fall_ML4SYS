import os

def load_ids_from_logs(logs_dir):
    """
    Load all unique IDs from the log files under the given directory.

    :param logs_dir: Directory containing the log files.
    :return: A set of IDs found in the log files under this directory.
    """
    ids_set = set()

    # Loop through all files in the logs directory
    for filename in os.listdir(logs_dir):
        if filename.endswith("_improving_passes.log"):
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, 'r') as file:
                    # Extract IDs from the log file (assuming IDs are in the lines of the log)
                    for line in file:
                        ids_set.add(line.strip())  # Assuming the ID is on each line
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
    
    return ids_set

def find_overlapping_ids(base_dir):
    """
    Find overlapping IDs between the `ir_{id}` folders in the base directory.

    :param base_dir: Base directory containing the `ir_{id}` folders.
    :return: A dictionary where keys are folder names and values are the set of overlapping IDs with all other folders.
    """
    # Get all `ir_{id}` folders
    folder_names = [f for f in os.listdir(base_dir) if f.startswith('ir_') and os.path.isdir(os.path.join(base_dir, f))]
    folder_ids = {}

    # Load the IDs from each folder
    for folder in folder_names:
        folder_path = os.path.join(base_dir, folder)
        folder_ids[folder] = load_ids_from_logs(folder_path)

    # Find overlaps between the folder IDs
    overlaps = {}
    for folder in folder_names:
        current_ids = folder_ids[folder]
        overlap_count = {}
        for other_folder in folder_names:
            if folder != other_folder:
                overlap = current_ids & folder_ids[other_folder]  # Set intersection
                overlap_count[other_folder] = len(overlap)
        overlaps[folder] = overlap_count

    return overlaps

def print_overlapping_info(overlaps):
    """
    Print the number of overlapping IDs for each folder with others.

    :param overlaps: A dictionary containing the overlapping IDs count between folders.
    """
    for folder, overlap_count in overlaps.items():
        print(f"Folder '{folder}' overlaps with:")
        for other_folder, count in overlap_count.items():
            print(f"  - {other_folder}: {count} overlapping IDs")
        print()

def main():
    base_dir = './ir/improving_passes_ir2vec'  # Replace with your base directory

    # Find overlaps
    overlaps = find_overlapping_ids(base_dir)
    
    # Print overlapping information
    print_overlapping_info(overlaps)

if __name__ == "__main__":
    main()    