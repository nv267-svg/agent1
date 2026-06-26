import pandas as pd

# Load the CSV file
df = pd.read_csv('one_row_per_cow_per_day.csv')

# Drop columns where ALL values are empty/NaN
df_cleaned = df.dropna(axis='columns', how='all')

# Save the result to a new CSV file
df_cleaned.to_csv('output.csv', index=False)

