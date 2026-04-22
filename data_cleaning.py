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

# Pivot Series Name into columns so each variable has its own column
data_wide = data_melted.pivot_table(
    index=['Country Name', 'Year'],
    columns='Series Name',
    values='Value'
).reset_index()
data_wide.columns.name = None

# Rename carbon emissions column for convenience (adjust to exact name in your data)
carbon_col = [col for col in data_wide.columns if 'carbon' in col.lower() or 'CO2' in col or 'emission' in col.lower()]
print("Carbon column found:", carbon_col)  # verify before renaming
data_wide = data_wide.rename(columns={carbon_col[0]: 'carbon_emissions'})

# Drop rows where target variable is null
data_cleaned = data_wide.dropna(subset=['carbon_emissions'])

# drop all null values in the dataset
data_cleaned = data_cleaned.dropna()

print(data_cleaned.head())
print(data_cleaned.shape)

# Save the cleaned data to a new CSV file
data_cleaned.to_csv('data_cleaned.csv', index=False)