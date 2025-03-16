import pandas as pd
import numpy as np
from ast import literal_eval

# Read the original CSV file
data = pd.read_csv('embedding.csv')

# Define a function to parse the embedding vector
def parse_embedding(x):
    try:
        return np.array(literal_eval(x))
    except ValueError:
        # Handle special cases like 'nan', 'inf', '-inf'
        x = x.strip('[]')
        nums = []
        for n in x.split(','):
            n = n.strip()
            if n.lower() == 'nan':
                nums.append(np.nan)
            elif n.lower() == 'inf':
                nums.append(np.inf)
            elif n.lower() == '-inf':
                nums.append(-np.inf)
            else:
                nums.append(float(n))
        return np.array(nums)

# Apply the function to the 'embedding_vector' column
data['embedding_vector'] = data['embedding_vector'].apply(parse_embedding)
# Convert NumPy arrays to lists
data['embedding_vector'] = data['embedding_vector'].apply(lambda x: x.tolist())

# Save to a new CSV file
data.to_csv('processed_file.csv', index=False)

# Step 1: Read the CSV file
data = pd.read_csv('processed_file.csv')

# Step 2: Define the parsing function
def parse_or_none(x):
    try:
        # Attempt to parse the string representation of the list
        return np.array(literal_eval(x))
    except Exception:
        print("Exception during parsing!")
        return None

# Step 3: Apply the parsing function to 'embedding_vector'
data['embedding_vector'] = data['embedding_vector'].apply(parse_or_none)

print(f"Number of entries after parsing: {len(data)}")

# Step 4: Drop rows where parsing failed (i.e., 'embedding_vector' is None)
data = data.dropna(subset=['embedding_vector']).reset_index(drop=True)

# Step 5: Define a function to check for NaN or inf in the embedding vector
def is_valid_vector(x):
    # Check if the vector contains any NaN or inf values
    return not (np.isnan(x).any() or np.isinf(x).any())

# Step 6: Filter out rows where 'embedding_vector' contains NaN or inf
data = data[data['embedding_vector'].apply(is_valid_vector)].reset_index(drop=True)

print(f"Number of entries after removing NaN/inf vectors: {len(data)}")

# Step 7: Define a function to check for zero vectors
def is_non_zero_vector(x):
    # Check if the vector is not all zeros
    return not np.all(x == 0)

# Step 8: Filter out rows where 'embedding_vector' is a zero vector
data = data[data['embedding_vector'].apply(is_non_zero_vector)].reset_index(drop=True)

print(f"Number of entries after removing zero vectors: {len(data)}")

# Step 9: Convert 'embedding_vector' NumPy arrays to lists for CSV compatibility
data['embedding_vector'] = data['embedding_vector'].apply(lambda x: x.tolist())

# Step 10: Save the cleaned DataFrame to a new CSV file
data.to_csv('processed_file.csv', index=False)
