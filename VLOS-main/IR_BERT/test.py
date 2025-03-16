import pandas as pd
import numpy as np

# Read your CSV file
data = pd.read_csv('embedding_vectors.csv')

# Now, 'embedding_vector' contains NumPy arrays
print(data.head())
data.head().to_csv('test.csv')
