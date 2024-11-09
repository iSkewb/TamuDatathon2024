import pandas as pd
from itertools import combinations

# Load the dataset
df = pd.read_csv("../data/data_cleaned.csv")

# Filter rows where 'Group Name' is not NaN
df = df.dropna(subset=['Word', 'Group Name'])

pairs = []
labels = []

# Iterate over each game ID and group words into pairs
for _, game_group in df.groupby('Game ID'):
    words = game_group['Word'].tolist()
    groups = game_group['Group Name'].tolist()
    
    # Create positive pairs (same group)
    for i, j in combinations(range(len(words)), 2):
        if groups[i] == groups[j]:
            pairs.append((words[i], words[j]))
            labels.append(1)  # Same group, label as 1
    
    # Create negative pairs (different groups)
    for i, j in combinations(range(len(words)), 2):
        if groups[i] != groups[j]:
            pairs.append((words[i], words[j]))
            labels.append(0)  # Different groups, label as 0

# Convert pairs to dataframe
pairs_df = pd.DataFrame({'word1': [pair[0] for pair in pairs], 'word2': [pair[1] for pair in pairs], 'label': labels})

pairs_df.to_csv("../data/pairs_data.csv", index=False)

print(pairs_df.head(20))