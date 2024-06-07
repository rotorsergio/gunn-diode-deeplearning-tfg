import pandas as pd
import pickle as pkl # pickle is a module that serializes python objects

exit_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/exit.csv'
normalized_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/normalized.csv'
standardized_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/standardized.csv'
values_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/values.pkl'

df = pd.read_csv(exit_path)
df = df.drop(columns=['Intervalley', 'FMA'])

mean = df.mean()
std = df.std()
max = df.max()
min = df.min()

# Save the values
with open(values_path, 'wb') as f:
    pkl.dump({'mean': mean, 'std': std, 'max': max, 'min': min}, f)

# Standardize and normalize the data (checking which is best later)
std_df = (df - mean) / std
norm_df = (df - min) / (max - min)

norm_df.to_csv(normalized_path, index=False)
std_df.to_csv(standardized_path, index=False)