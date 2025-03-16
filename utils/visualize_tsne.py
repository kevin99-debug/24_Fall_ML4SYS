import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = pd.read_csv("embedding_vectors.csv")

# Convert embedding vectors from string representation to NumPy arrays
vectors = data['embedding_vector'].apply(lambda x: np.array(literal_eval(x))).tolist()
vectors = np.array(vectors)

# Apply t-SNE with 3 components for 3D plotting
tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
reduced_vectors = tsne.fit_transform(vectors)

# Plot in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], alpha=0.5, c=reduced_vectors[:, 2], cmap='viridis', s=0.2)
ax.set_title("t-SNE 3D Visualization of Program Vectors")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
fig.colorbar(scatter, ax=ax, label="Dimension 3 Color Scale")
plt.savefig("program_vector_tsne_3d.png")
plt.show()
