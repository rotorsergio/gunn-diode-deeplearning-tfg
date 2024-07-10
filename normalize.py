import pandas as pd
import pickle as pkl # pickle is a module that serializes python objects

exit_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/exit.csv'
reduced_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/200nm_fine.csv'

normalized_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/normalized.csv'
standardized_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/standardized.csv'
norm_reduced_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/norm_200nm_fine.csv'
values_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/values.pkl'
fine_values_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/fine_values.pkl'

df = pd.read_csv(exit_path)
df = df.drop(columns=['Intervalley'])

mean = df.mean()
std = df.std()
max = df.max()
min = df.min()

print('Mean:\n', mean)
print('Standard deviation:\n', std)
print('Maximum:\n', max)
print('Minimum:\n', min)

# Save the values
with open(values_path, 'wb') as f:
    pkl.dump({'mean': mean, 'std': std, 'max': max, 'min': min}, f)

# Load the values FOR LATER USE
"""
with open(values_path, 'rb') as f:
    values = pkl.load(f)
    mean = values['mean']
    std = values['std']
    max = values['max']
    min = values['min']

# Denormalize the data
df = df * std + mean
# Destandardize the data
df = df * (max - min) + min
"""

# Standardize and normalize the data (checking which is best later)
# std_df = (df - mean) / std
norm_df = (df - min) / (max - min)
# norm_df['Wo'] = 0

# norm_df.to_csv(norm_reduced_path, index=False)
norm_df.to_csv(normalized_path, index=False)
# std_df.to_csv(standardized_path, index=False)