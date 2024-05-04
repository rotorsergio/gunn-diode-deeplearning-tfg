import pandas as pd
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

# Read the file
df = pd.read_fwf('CORR01')

# Select the 5th column (assuming the first column is at index 0)
df = df.iloc[:, 4]

# Write the selected column to a new file
df.to_csv('corr_reduced', index=False)

os.remove('CORR01')