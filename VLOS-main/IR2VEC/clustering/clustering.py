import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import csv

# Load the data
data = pd.read_csv("embedding.csv")

# Convert embedding vectors from string representation to NumPy arrays
vectors = data['embedding_vector'].apply(lambda x: np.array(literal_eval(x))).tolist()
vectors = np.array(vectors)

# Apply K-means clustering on the original high-dimensional vectors
num_clusters = 8  # Set the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=10)
labels = kmeans.fit_predict(vectors)

# Save the cluster centers to a CSV file
cluster_centers = kmeans.cluster_centers_
with open("kmeans_cluster_centers.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['embedding_vector'])
    for cluster_center in cluster_centers:
      embedding_str = ','.join([f"{x:.6f}" for x in cluster_center])
      writer.writerow([embedding_str])

# Apply t-SNE for 2D visualization on the original high-dimensional vectors
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
reduced_vectors = tsne.fit_transform(vectors)

# 2D Plot with K-means Clusters from High-Dimensional Data
plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)
plt.title("t-SNE 2D Visualization with K-Means Clusters")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(scatter, label="Cluster Label")
plt.savefig("program_vector_kmeans_tsne_2d.png")
plt.show()
