import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

# Identify the year columns (they start with a 4-digit year)
year_cols = [col for col in data.columns if col[:4].isdigit()]

# Melt year columns into rows
data_melted = data.melt(
    id_vars=['Country Name', 'Series Name'],
    value_vars=year_cols,
    var_name='Year',
    value_name='Value'
)

# Extract just the numeric year from strings like "1976 [YR1976]"
data_melted['Year'] = data_melted['Year'].str.extract(r'(\d{4})').astype(int)

# Filter to 2001–2024
data_melted = data_melted[data_melted['Year'].between(2001, 2024)]

# Replace ".." (missing values in World Bank data) with NaN
data_melted['Value'] = pd.to_numeric(data_melted['Value'], errors='coerce')

print(data_melted.head())


# Drop rows with missing values in 'Value' column
data_cleaned = data_melted.dropna(subset=['Value'])

# Save the cleaned data to a new CSV file
data_cleaned.to_csv('data_cleaned.csv', index=False)