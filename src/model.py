import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from itertools import combinations

# Load and process data
df = pd.read_csv("../data/data_cleaned.csv")

# Ensure there are no NaN values in the relevant columns
df = df.dropna(subset=['Word', 'Group Name'])

# Sample 256 rows for faster training
df_sample = df.sample(n=256, random_state=42)

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

# Predict pairs of new words
pair_predictions = {}
for (i, j) in combinations(range(len(new_words)), 2):
    pair = (new_words[i], new_words[j])
    features = np.hstack((new_embeddings[i], new_embeddings[j])).reshape(1, -1)
    
    # Optionally reduce dimensionality for new word pairs (same PCA transformation)
    features_reduced = pca.transform(features)
    
    prediction = clf.predict(features_reduced)
    pair_predictions[pair] = prediction

# Output pair predictions
print(pair_predictions)