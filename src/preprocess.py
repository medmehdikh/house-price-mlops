import pandas as pd
import os

# Create processed directory if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# Your actual data preprocessing code here
df = pd.read_csv('data/train.csv')
# ... do preprocessing ...
df.to_csv('data/processed/train.csv', index=False)
print("Data preprocessing completed!")