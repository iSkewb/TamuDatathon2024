import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

# Load data
data = pd.read_csv("../data/pairs_data.csv")

# Initialize SentenceTransformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare embeddings for each word
def get_word_embedding(word):
    return model.encode(word)

# Create custom dataset class
class WordPairDataset(Dataset):
    def __init__(self, word_pairs, labels):
        self.word_pairs = word_pairs
        self.labels = labels

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx):
        word1, word2 = self.word_pairs[idx]
        label = self.labels[idx]
        # Get embeddings for each word
        word1_embedding = torch.tensor(get_word_embedding(word1), dtype=torch.float32)
        word2_embedding = torch.tensor(get_word_embedding(word2), dtype=torch.float32)
        # Concatenate word embeddings
        features = torch.cat((word1_embedding, word2_embedding), dim=0)
        return features, torch.tensor(label, dtype=torch.float32)

# Prepare the data
word_pairs = list(zip(data['word1'], data['word2']))
labels = data['label'].tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(word_pairs, labels, test_size=0.2, random_state=42)

# Create DataLoader for batching
train_dataset = WordPairDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = WordPairDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
class WordPairModel(nn.Module):
    def __init__(self, embedding_size=384):
        super(WordPairModel, self).__init__()
        self.fc1 = nn.Linear(embedding_size * 2, 512)  # 2 * embedding size
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

# Instantiate the model, loss function, and optimizer
model_nn = WordPairModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model_nn.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model_nn(inputs)
        loss = criterion(outputs.flatten(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Evaluate the model
model_nn.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model_nn(inputs)
        predicted = (outputs.flatten() > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
