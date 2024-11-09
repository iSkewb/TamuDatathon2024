import pandas as pd

# Load the dataset
df = pd.read_csv("../../data/pairs_data.csv")  # Adjust the path to your CSV file

# Check column names to confirm
print("Column names:", df.columns)

# Remove leading/trailing spaces from column names if necessary
df.columns = df.columns.str.strip()

# Drop rows with NaN values in 'Word' and 'Group Name' columns
if 'Word' in df.columns and 'Group Name' in df.columns:
    df = df.dropna(subset=['Word', 'Group Name'])
else:
    print("Columns 'Word' and 'Group Name' are missing.")