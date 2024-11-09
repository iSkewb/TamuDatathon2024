import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from k_means_constrained import KMeansConstrained
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load and process data
df = pd.read_csv("../../data/data_cleaned.csv")

# Ensure there are no NaN values in the relevant columns
df = df.dropna(subset=['Word', 'Group Name'])

# Sample data for faster training (You can change this as per your requirement)
df_sample = df.sample(n=8257, random_state=42)

# Pre-trained embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Prepare pairwise data for training (optional)
pairs = []
labels = []

# Grouping by 'Game ID' from the sampled data
for _, game_group in df_sample.groupby('Game ID'):
    words = game_group['Word'].tolist()
    groups = game_group['Group Name'].tolist()
    # print(len(groups))
    
    for (i, j) in combinations(range(len(words)), 2):
        word_pair = (words[i], words[j])
        label = 1 if groups[i] == groups[j] else 0
        pairs.append(word_pair)
        labels.append(label)

# Ensure all pairs are strings
pairs = [(str(pair[0]), str(pair[1])) for pair in pairs]
# print(len(pairs))

# Batch encoding of word pairs to speed up the process
print([pairs[0][0]] + [pairs[0][1]])
word_embeddings = model.encode([pair[0] for pair in pairs] + [pair[1] for pair in pairs], batch_size=32)
print("Number of embeddings:", len(word_embeddings))
print(word_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(word_embeddings[:, 0], word_embeddings[:, 1], s=50, cmap='viridis')
plt.title("2D Representation of Embedding Similarity (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Create feature matrix X by concatenating embeddings for each pair (optional)
X = np.array([np.hstack((word_embeddings[i], word_embeddings[i + len(pairs)])) for i in range(len(pairs))])
# print(len(X))
y = np.array(labels)

# Optionally reduce the dimensionality of the embeddings (e.g., using PCA)
pca = PCA(n_components=50)  # Reduce to 50 components
X_reduced = pca.fit_transform(X)

# Split data into training and testing sets (optional)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Train a classifier (optional for evaluation)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# To predict groups for new words:
new_words = ["BENT", "GNARLY", "TWISTED", "WARPED", "LICK", "OUNCE", "SHRED", "TRACE", "EXPONENT", "POWER", "RADICAL", "ROOT", "BATH", "POWDER", "REST", "THRONE"]

# Ensure new words are strings
new_words = [str(word) for word in new_words]

# Get embeddings for new words (in batch)
new_embeddings = model.encode(new_words, batch_size=32)

# plt.figure(figsize=(8, 6))
# plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1], s=50, cmap='viridis')
# plt.title("2D Representation of Embedding Similarity (PCA)")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.show()

# Calculate pairwise cosine similarities
cosine_similarities = np.zeros((len(new_words), len(new_words)))

for i in range(len(new_words)):
    for j in range(i + 1, len(new_words)):
        cosine_similarities[i, j] = 1 - cosine(new_embeddings[i], new_embeddings[j])
        cosine_similarities[j, i] = cosine_similarities[i, j]

# Use KMeans clustering to group words into 4 clusters based on similarity
clf = KMeansConstrained(n_clusters=4, size_min=4, size_max=4, random_state=42)
clusters = clf.fit_predict(cosine_similarities)
print(clusters)

# Ensure the clusters are valid (4 groups with 4 words each)
assert len(set(clusters)) == 4, "Clusters should have exactly 4 groups."
assert all(np.sum(clusters == cluster_id) == 4 for cluster_id in range(4)), "Each cluster should have exactly 4 words."

# Arrange words into 4x4 matrix by cluster
matrix = np.full((4, 4), None, dtype=object)  # Initialize with None or any placeholder

# Shuffle the words within clusters for better diversity (optional)
shuffled_words = [new_words[i] for i in range(len(new_words))]
np.random.shuffle(shuffled_words)

# Fill the matrix with the shuffled words
row, col = 0, 0
for word in shuffled_words:
    matrix[row, col] = word
    col += 1
    if col == 4:
        col = 0
        row += 1

# Output the 4x4 matrix of related words
print("4x4 Word Similarity Matrix:")
print(matrix)
