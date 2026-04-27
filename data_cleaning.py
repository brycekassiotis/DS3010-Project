import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

# Rename carbon emissions column for convenience
carbon_col = [col for col in data_wide.columns if 'carbon' in col.lower() or 'CO2' in col or 'emission' in col.lower()]
print("Carbon column found:", carbon_col)
data_wide = data_wide.rename(columns={carbon_col[0]: 'carbon_emissions'})

cols_to_drop = [
    'GDP (current LCU)',
    'Access to electricity, rural (% of rural population)',
    'Electricity production from oil, gas and coal sources (% of total)',
    'Urban population (% of total population)',
]
data_wide = data_wide.drop(columns=cols_to_drop)

# Drop rows where target variable is null
data_cleaned = data_wide.dropna(subset=['carbon_emissions'])

# drop all null values in the dataset
data_cleaned = data_cleaned.dropna()

print(data_cleaned.head())
print(data_cleaned.shape)

# save full cleaned dataset for later reference
data_cleaned.to_csv('pre_split.csv', index=False)

# Split by country so no country appears in more than one split
countries = data_cleaned['Country Name'].unique()
train_countries, temp_countries = train_test_split(countries, test_size=0.4, random_state=42)
val_countries, test_countries = train_test_split(temp_countries, test_size=0.5, random_state=42)

train_data = data_cleaned[data_cleaned['Country Name'].isin(train_countries)]
val_data = data_cleaned[data_cleaned['Country Name'].isin(val_countries)]
test_data = data_cleaned[data_cleaned['Country Name'].isin(test_countries)]

X_train = train_data.drop(columns=['carbon_emissions'])
y_train = train_data['carbon_emissions']
X_val = val_data.drop(columns=['carbon_emissions'])
y_val = val_data['carbon_emissions']
X_test = test_data.drop(columns=['carbon_emissions'])
y_test = test_data['carbon_emissions']

print(f"Train: {len(train_countries)} countries, {len(train_data)} rows")
print(f"Val:   {len(val_countries)} countries, {len(val_data)} rows")
print(f"Test:  {len(test_countries)} countries, {len(test_data)} rows")

# Save all splits
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)