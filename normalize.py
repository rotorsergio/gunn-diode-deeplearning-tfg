import pandas as pd

exit_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/exit.csv'
normalized_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/normalized.csv'

df = pd.read_csv(exit_path)
df = df.drop(columns=['Intervalley', 'FMA'])

# Standarize the data
df = (df - df.mean()) / df.std()

df.to_csv(normalized_path, index=False)