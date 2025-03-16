import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the CSV file
csv_file = "embedding_vectors.csv"  # Replace with your actual file name
df = pd.read_csv(csv_file)

# Extract the embedding vectors as a NumPy array
# Assuming 'embedding vector' column contains lists of embeddings
df['embedding_vector'] = df['embedding_vector'].apply(eval)  # Convert strings to lists if needed
embeddings = np.stack(df['embedding_vector'].values)  # Shape: (num_vectors, embedding_dim)

# Calculate cosine similarity between each pair of vectors
similarity_matrix = cosine_similarity(embeddings)

# Find the most similar vector for each row
most_similar_indices = np.argmax(similarity_matrix - np.eye(len(embeddings)), axis=1)  # Exclude self-similarity
most_similar_scores = similarity_matrix[np.arange(len(embeddings)), most_similar_indices]

# Add results to the DataFrame
df['most_similar_index'] = most_similar_indices
df['most_similar_score'] = most_similar_scores

# Print the DataFrame with the most similar vector for each row
print(df[['ir_id', 'loop_id', 'header', 'most_similar_index', 'most_similar_score']])
