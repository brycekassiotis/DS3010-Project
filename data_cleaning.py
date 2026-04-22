import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

# Extract the 4-digit year from strings like "1984 [YR1984]"
data['Year'] = data['Year'].str.extract(r'(\d{4})').astype(float)

# Now filter to 2001–2024
data['Year'] = data['Year'].apply(lambda x: x if 2001 <= x <= 2024 else np.nan)

