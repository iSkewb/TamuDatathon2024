import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from random import choice

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example list of 16 words (replace with your actual list)
words = ["BENT", "GNARLY", "TWISTED", "WARPED", "LICK", "OUNCE", "SHRED", "TRACE",
         "EXPONENT", "POWER", "RADICAL", "ROOT", "BATH", "POWDER", "REST", "THRONE"]

# Get embeddings for all words
word_embeddings = model.encode(words)

# Precompute cosine similarity matrix (NxN matrix where N is number of words)
cosine_sim_matrix = np.zeros((len(words), len(words)))
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        cosine_sim = 1 - cosine(word_embeddings[i], word_embeddings[j])
        cosine_sim_matrix[i, j] = cosine_sim
        cosine_sim_matrix[j, i] = cosine_sim  # Symmetric matrix

# Function to create a cluster from a starting word
def create_cluster(start_idx, embeddings, similarity_matrix):
    cluster = [start_idx]
    cluster_center = embeddings[start_idx]

    # Store which words have already been assigned to a cluster
    assigned_words = {start_idx}

    while len(cluster) < 4:
        # Find the word closest to the current cluster center
        cluster_similarities = similarity_matrix[cluster].mean(axis=0)
        most_similar_idx = np.argmax(cluster_similarities)
        
        if most_similar_idx not in assigned_words:
            cluster.append(most_similar_idx)
            assigned_words.add(most_similar_idx)

    return cluster

# Main loop to form 4 clusters
clusters = []
remaining_words = list(range(len(words)))

while len(remaining_words) > 0:
    # Randomly pick a word to start the cluster
    start_idx = choice(remaining_words)
    cluster = create_cluster(start_idx, word_embeddings, cosine_sim_matrix)
    
    # Remove the words from the remaining list
    for idx in cluster:
        remaining_words.remove(idx)
    
    clusters.append(cluster)

# After clustering, we want to calculate the "cohesiveness" of each cluster
def calculate_cluster_cohesiveness(cluster, embeddings, similarity_matrix):
    """Calculate the cohesiveness of a cluster by finding the average distance of each word to the cluster center"""
    cluster_center = np.mean([embeddings[i] for i in cluster], axis=0)
    cohesiveness = 0
    for i in cluster:
        cohesiveness += similarity_matrix[i, cluster_center]
    return cohesiveness

# Calculate cohesiveness for each cluster
cohesiveness_scores = [calculate_cluster_cohesiveness(cluster, word_embeddings, cosine_sim_matrix) for cluster in clusters]

# Get the index of the most cohesive cluster (lowest score)
best_cluster_idx = np.argmin(cohesiveness_scores)

# Print the final clusters and the best one
print("Final clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {[words[i] for i in cluster]}")

print(f"The most cohesive cluster is cluster {best_cluster_idx + 1} with words: {[words[i] for i in clusters[best_cluster_idx]]}")
