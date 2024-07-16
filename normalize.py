import pandas as pd
import pickle as pkl # pickle is a module that serializes python objects

exit_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/exit.csv'
reduced_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/200nm.csv'

normalized_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/normalized.csv'
norm_reduced_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/norm_200nm.csv'
values_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/values.pkl'
fine_values_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/fine_values.pkl'

datamode = 'reduced' # Modes: 'exit', 'reduced'

if datamode == 'exit':
    df = pd.read_csv(exit_path)
    df = df.drop(columns=['Intervalley'])
    max = df.max()
    min = df.min()
    with open(values_path, 'wb') as f:
        pkl.dump({'max': max, 'min': min}, f)
    norm_df = (df - min) / (max - min)
    print(norm_df)
    norm_df.to_csv(normalized_path, index=False)
elif datamode == 'reduced':
    df = pd.read_csv(reduced_path)
    df = df.drop(columns=['Wo'])
    max = df.max()
    min = df.min()
    with open(fine_values_path, 'wb') as f:
        pkl.dump({'max': max, 'min': min}, f)
    norm_df = (df - min) / (max - min)
    print(norm_df)
    norm_df.to_csv(norm_reduced_path, index=False)
else:
    raise ValueError('Invalid data mode')