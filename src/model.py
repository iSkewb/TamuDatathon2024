import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import combinations

# Load and process data
df = pd.read_csv("../data/data_cleaned.csv")

# Pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare pairwise data for training
pairs = []
labels = []

for _, game_group in df.groupby('Game ID'):
    words = game_group['Word'].tolist()
    groups = game_group['Group Name'].tolist()
    
    for (i, j) in combinations(range(len(words)), 2):
        word_pair = (words[i], words[j])
        label = 1 if groups[i] == groups[j] else 0
        pairs.append(word_pair)
        labels.append(label)

# Ensure all pairs are strings and handle possible NaN values
pairs = [(str(pair[0]), str(pair[1])) for pair in pairs]

# Get embeddings for each word pair
X = np.array([np.hstack((model.encode(pair[0]), model.encode(pair[1]))) for pair in pairs])
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# To predict groups for new words:
new_words = ["BENT", "GNARLY", "TWISTED", "WARPED", "LICK", "OUNCE", "SHRED", "TRACE", "EXPONENT", "POWER", "RADICAL", "ROOT", "BATH", "POWDER", "REST", "THRONE"]  # example set of 16 words

# Ensure new words are strings
new_words = [str(word) for word in new_words]

# Get embeddings for new words
new_embeddings = model.encode(new_words)

pair_predictions = {}
for (i, j) in combinations(range(len(new_words)), 2):
    pair = (new_words[i], new_words[j])
    features = np.hstack((new_embeddings[i], new_embeddings[j])).reshape(1, -1)
    prediction = clf.predict(features)
    pair_predictions[pair] = prediction

# Output pair predictions
print(pair_predictions)
