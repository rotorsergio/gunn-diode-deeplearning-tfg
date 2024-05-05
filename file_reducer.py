import os
import pandas as pd
from tqdm import tqdm # for progress bar

# Specify the parent directory
parent_dir = 'D:/datos_montecarlo'

# Get a list of all 'CORR01' files
files = [os.path.join(dirpath, file)
         for dirpath, dirnames, filenames in os.walk(parent_dir)
         for file in filenames if file == 'CORR01']


# Walk through all subdirectories
for file_path in tqdm(files):
    # Read the file
    df = pd.read_fwf(file_path)

    # Select the 5th column (assuming the first column is at index 0)
    df = df.iloc[:, 4]

    # Write the selected column to a new file
    reduced_file_path = os.path.join(os.path.dirname(file_path), 'corr_reduced')
    df.to_csv(reduced_file_path, index=False)

    # Erase the original 'CORR01' file
    os.remove(file_path)