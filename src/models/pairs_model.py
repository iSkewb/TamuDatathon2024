import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K
from itertools import combinations

# Load your dataset
df = pd.read_csv("../data/pairs_data.csv")  # Use the correct path to your dataset

# Data preprocessing: Remove rows with NaN values
df = df.dropna(subset=['Word', 'Group Name'])

# Generate word pairs with labels
pairs = []
labels = []

# Group by 'Game ID' and generate word pairs within the same game
for _, game_group in df.groupby('Game ID'):
    words = game_group['Word'].tolist()
    groups = game_group['Group Name'].tolist()
    
    # Create pairs and labels
    for i, j in combinations(range(len(words)), 2):
        pair = (words[i], words[j])
        label = 1 if groups[i] == groups[j] else 0
        pairs.append(pair)
        labels.append(label)

# Convert pairs and labels to DataFrame for easier handling
pairs_df = pd.DataFrame({'word1': [pair[0] for pair in pairs], 'word2': [pair[1] for pair in pairs], 'label': labels})

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to encode pairs of words into embeddings
def encode_pairs(pairs):
    word1_embeddings = model.encode([pair[0] for pair in pairs], show_progress_bar=True)
    word2_embeddings = model.encode([pair[1] for pair in pairs], show_progress_bar=True)
    
    return np.array(word1_embeddings), np.array(word2_embeddings)

# Encode the word pairs into embeddings
word1_embeddings, word2_embeddings = encode_pairs(zip(pairs_df['word1'], pairs_df['word2']))

# Define the Siamese Network Model
def siamese_model(input_shape):
    # Input layers for two words
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    # Use a simple feed-forward network for each word pair
    dense = Dense(128, activation='relu')(input_1)
    dense = Dropout(0.2)(dense)
    dense = Dense(128, activation='relu')(dense)

    # Lambda layer to compute the absolute difference between the embeddings
    diff = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([dense, dense])

    # Final layer to predict similarity (0 or 1)
    output = Dense(1, activation='sigmoid')(diff)

    model = Model(inputs=[input_1, input_2], outputs=output)

    return model

# Create the model
input_shape = word1_embeddings.shape[1:]
model = siamese_model(input_shape)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train-test split
X_train_1, X_test_1, X_train_2, X_test_2, y_train, y_test = train_test_split(word1_embeddings, word2_embeddings, pairs_df['label'], test_size=0.2, random_state=42)

# Train the model
history = model.fit([X_train_1, X_train_2], y_train, epochs=10, batch_size=64, validation_data=([X_test_1, X_test_2], y_test))

# Evaluate the model
loss, accuracy = model.evaluate([X_test_1, X_test_2], y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Make predictions for new word pairs
new_pairs = [("SNOW", "HAIL"), ("SNOW", "LEVEL"), ("SHIFT", "KAYAK")]

# Encode new word pairs
new_word1_embeddings, new_word2_embeddings = encode_pairs(new_pairs)

# Predict similarity (0 or 1)
predictions = model.predict([new_word1_embeddings, new_word2_embeddings])

print("Predictions (0 or 1, dissimilar or similar):")
print(predictions)
