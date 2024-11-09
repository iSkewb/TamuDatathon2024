from gensim.models import Word2Vec
from gensim.models import common_texts
from nltk.tokenize import word_tokenize, sent_tokenize

model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

words = ["BENT", "GNARLY", "TWISTED", "WARPED", "LICK", "OUNCE", "SHRED", "TRACE", "EXPONENT", "POWER", "RADICAL", "ROOT", "BATH", "POWDER", "REST", "THRONE"]

from sklearn.cluster import KMeans
import numpy as np

# Generate vectors for each word in the list
word_vectors = np.array([model.wv[word] for word in words if word in model.wv])

# Apply KMeans clustering to group similar words
num_clusters = 4  # Assumed number of clusters in the Connections game
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(word_vectors)

# Print each cluster of words
for i in range(num_clusters):
    cluster_words = [words[j] for j in range(len(words)) if kmeans.labels_[j] == i]
    print(f"Cluster {i+1}: {cluster_words}")
