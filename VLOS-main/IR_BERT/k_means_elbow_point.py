import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
data = pd.read_csv("embedding_vectors.csv")

# Convert embedding vectors from string representation to NumPy arrays
vectors = data['embedding_vector'].apply(lambda x: np.array(literal_eval(x))).tolist()
vectors = np.array(vectors)

# Normalize the vectors for cosine similarity
vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Compute K-means inertia for clusters ranging from 1 to 100
inertia_values = []
cluster_range = range(1, 101)

for k in cluster_range:
    # Use KMeans with cosine similarity
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(vectors_normalized)
    
    # Calculate the inertia based on cosine distance
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Compute cosine distance inertia
    inertia = 0
    for i in range(k):
        cluster_vectors = vectors_normalized[labels == i]
        center = cluster_centers[i]
        distances = 1 - cosine_similarity(cluster_vectors, [center]).flatten()  # Cosine distance = 1 - similarity
        inertia += np.sum(distances ** 2)
    
    inertia_values.append(inertia)

# Save inertia values to a CSV file
inertia_df = pd.DataFrame({"Number of Clusters (k)": list(cluster_range), "Inertia": inertia_values})
inertia_df.to_csv("kmeans_inertia_values_cosine.csv", index=False)

# Plot the inertia values
plt.figure(figsize=(10, 7))
plt.plot(cluster_range, inertia_values, marker='o', linestyle='--', color='b')
plt.title("Elbow Method with Cosine Distance for Determining Optimal Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Cosine Distance)")
plt.grid()
plt.savefig("kmeans_elbow_point_cosine.png")
plt.show()
