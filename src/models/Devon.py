from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np

# Train the Word2Vec model
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

# List of words to cluster
words = ["BENT", "GNARLY", "TWISTED", "WARPED", "LICK", "OUNCE", "SHRED", "TRACE", 
         "EXPONENT", "POWER", "RADICAL", "ROOT", "BATH", "POWDER", "REST", "THRONE"]

# Filter words to include only those in the Word2Vec vocabulary
filtered_words = [word for word in words if word in model.wv]

print(common_texts)

# Check if filtered_words is not empty
if filtered_words:
    # Generate vectors for each word in the filtered list
    word_vectors = np.array([model.wv[word] for word in filtered_words])

    # Apply KMeans clustering to group similar words
    num_clusters = 4  # Assumed number of clusters in the Connections game
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(word_vectors)

    # Print each cluster of words
    for i in range(num_clusters):
        cluster_words = [filtered_words[j] for j in range(len(filtered_words)) if kmeans.labels_[j] == i]
        print(f"Cluster {i+1}: {cluster_words}")
else:
    print("No words in the list are found in the model vocabulary.")
