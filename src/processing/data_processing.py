import pandas as pd

# Read in the data
df = pd.read_csv("../data/Connections_Data.csv")

# Drop the 'Starting Row' and 'Starting Column' columns
df_cleaned = df.drop(columns=['Starting Row', 'Starting Column'])

# Optionally, save the cleaned data
df_cleaned.to_csv("../data/data_cleaned.csv", index=False)
