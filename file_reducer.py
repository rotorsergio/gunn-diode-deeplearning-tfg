import os
from tqdm import tqdm # for progress bar
from multiprocessing import Pool, cpu_count

# Redirect stdout (standard output) and stderr (standard error) to a file
# sys.stdout = open('output.txt', 'w')
# sys.stderr = sys.stdout

# Function to process a single file
def process_file(file_path):
    colspecs = [(53, 66)]
    # Read the file
    with open(file_path, 'r') as f:
        data = f.readlines()
    # Extract the 5th column
    column5 = [line[colspecs[0][0]:colspecs[0][1]] for line in data]
    # Write the selected column to a new file
    with open(file_path.replace('CORR01', 'corr_reduced'), 'w') as f:
        for item in column5:
            f.write("%s\n" % item)
    # Delete the original file
    os.remove(file_path)

# Protect the main part of the code
if __name__ == '__main__':
    # Specify the parent directory
    parent_dir = 'D:/datos_montecarlo'

    # Get a list of all 'CORR01' files
    files = [os.path.join(dirpath, file)
             for dirpath, dirnames, filenames in os.walk(parent_dir)
             for file in filenames if file == 'CORR01']

    # Create a multiprocessing Pool
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(process_file, files), total=len(files)))