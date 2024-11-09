import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from k_means_constrained import KMeansConstrained

# Load and process data
df = pd.read_csv("../data/data_cleaned.csv")

# Ensure there are no NaN values in the relevant columns
df = df.dropna(subset=['Word', 'Group Name'])

# Sample 256 rows for faster training
df_sample = df.sample(n=8257, random_state=42)

# Pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare pairwise data for training
pairs = []
labels = []

# Grouping by 'Game ID' from the sampled data
for _, game_group in df_sample.groupby('Game ID'):
    words = game_group['Word'].tolist()
    groups = game_group['Group Name'].tolist()
    
    for (i, j) in combinations(range(len(words)), 2):
        word_pair = (words[i], words[j])
        label = 1 if groups[i] == groups[j] else 0
        pairs.append(word_pair)
        labels.append(label)

# Ensure all pairs are strings
pairs = [(str(pair[0]), str(pair[1])) for pair in pairs]

# Batch encoding of word pairs to speed up the process
word_embeddings = model.encode([pair[0] for pair in pairs] + [pair[1] for pair in pairs], batch_size=32)

# Create feature matrix X by concatenating embeddings for each pair
X = np.array([np.hstack((word_embeddings[i], word_embeddings[i + len(pairs)])) for i in range(len(pairs))])
y = np.array(labels)

# Optionally reduce the dimensionality of the embeddings (e.g., using PCA) to speed up training
pca = PCA(n_components=50)  # Reduce to 50 components
X_reduced = pca.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Train a classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# To predict groups for new words:
new_words = ["BENT", "GNARLY", "TWISTED", "WARPED", "LICK", "OUNCE", "SHRED", "TRACE", "EXPONENT", "POWER", "RADICAL", "ROOT", "BATH", "POWDER", "REST", "THRONE"]

# Ensure new words are strings
new_words = [str(word) for word in new_words]

# Get embeddings for new words (in batch)
new_embeddings = model.encode(new_words, batch_size=32)

# Calculate pairwise cosine similarities
cosine_similarities = np.zeros((len(new_words), len(new_words)))

for i in range(len(new_words)):
    for j in range(i + 1, len(new_words)):
        cosine_similarities[i, j] = 1 - cosine(new_embeddings[i], new_embeddings[j])
        cosine_similarities[j, i] = cosine_similarities[i, j]

# Use KMeans clustering to group words into 4 clusters based on similarity
kmeans = KMeans(n_clusters=4, random_state=42)
# clusters = kmeans.fit_predict(cosine_similarities)
clf = KMeansConstrained(n_clusters=4, size_min=4, size_max=4, random_state=42)
clusters = clf.fit_predict(cosine_similarities)
print(clusters)

# Arrange words into 4x4 matrix by cluster
matrix = np.full((4, 4), None, dtype=object)  # Initialize with None or any placeholder

for cluster_id in range(4):
    cluster_words = [new_words[i] for i in range(len(new_words)) if clusters[i] == cluster_id]
    
    # Ensure each cluster fits into the matrix row with at most 4 words
    # If a cluster has more than 4 words, truncate it, or if fewer, fill with None
    cluster_words = cluster_words[:4]  # Truncate to 4 words if necessary
    matrix[cluster_id, :len(cluster_words)] = cluster_words

# Output the 4x4 matrix of related words
print("4x4 Word Similarity Matrix:")
print(matrix)
