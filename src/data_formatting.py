import pandas as pd

# Load the dataset
df = pd.read_csv("../data/Connections_Data.csv")

# Step 1: Sort by Game ID and Group Level to ensure consistency
df = df.sort_values(by=['Game ID', 'Group Level', 'Starting Row', 'Starting Column']).reset_index(drop=True)

# Step 2: Add a 'WordNumber' within each Game ID to identify position (1-16)
df['WordNumber'] = df.groupby('Game ID').cumcount() + 1

# Step 3: Pivot the data so each Game ID has one row with columns for each Word and Group Name
# Pivot words
words_pivot = df.pivot(index='Game ID', columns='WordNumber', values='Word')
words_pivot.columns = [f"Word{col}" for col in words_pivot.columns]  # Rename columns

# Get unique Group Names by grouping and joining
group_names = df.groupby(['Game ID', 'Group Level'])['Group Name'].first().unstack().reset_index()
group_names.columns = ['Game ID', 'Group1', 'Group2', 'Group3', 'Group4']  # Rename columns

# Step 4: Merge the word columns with group name columns
df_final = words_pivot.merge(group_names, on="Game ID")

# Save the final DataFrame to a new CSV
df_final.to_csv("../data/formatted_data.csv", index=False)

# Print the resulting DataFrame
print(df_final)
