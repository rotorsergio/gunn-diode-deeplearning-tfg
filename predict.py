import pandas as pd
import numpy as np
import pickle as pkl
import itertools
import keras

# ========== CONSTANTS ==========
desired_number_of_points: int = 21

norm_model_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model_norm.keras'
std_model_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model_std.keras'
fine_model_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model_finenorm.keras'

norm_prediction_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/norm_prediction.csv'
std_prediction_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/std_prediction.csv'
fine_prediction_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/fine_prediction.csv'

# ========== FUNCTIONS ==========

def generate_input_data():

    # Define the input data

    # Wo = np.arange(200, 360, 8) # last value target: 352
    Wo = np.arange(200, 356, 4)

    # Vds = np.arange(10, 65, 5) # last value target: 60
    Vds = np.arange(1, 61, 1)

    Temp = np.array([300, 400, 500])

    # Nd = np.array([0.5, 1, 5, 10])*1e24
    Nd = np.arange(0.5, 10.5, 0.5)*1e24

    #Create the input dataframe as an iteration of all the possible combinations of the input data
    combinations = itertools.product(Wo, Vds, Temp, Nd)
    # combinations = itertools.product(Vds, Temp, Nd)
    input_df = pd.DataFrame(combinations, columns=['Wo', 'Vds', 'Temp', 'Nd'])
    # input_df = pd.DataFrame(combinations, columns=['Vds', 'Temp', 'Nd'])
    # input_df['Wo'] = Wo

    return input_df

def get_values_from_pkl(values_path):
    with open(values_path, 'rb') as f:
        values = pkl.load(f)
        mean = values['mean']
        std = values['std']
        max = values['max']
        min = values['min']

    return mean, std, max, min

def normalize_data(data, max, min):
    data = (data - min) / (max - min)
    return data

def standardize_data(data, mean, std):
    data = (data - mean) / std
    return data

def prediction(input_df, model_path):
    model = keras.models.load_model(model_path)
    if isinstance(model, keras.models.Model): # This is done just to specify the code that model is a keras model
        # Ensure input_df is a numpy array
        if isinstance(input_df, pd.DataFrame):
            input_data = input_df.values
        else:
            input_data = input_df
        
        prediction = model.predict(input_data)
        return prediction
    else:
        raise ValueError("Failed to load the model.")

def denormalize_data(data, max, min):
    denorm = data * (max - min) + min
    denorm['Vds'] = denorm['Vds'].round(0)
    denorm['Nd'] = denorm['Nd'].apply(lambda x: round(x, 1-int(np.floor(np.log10(abs(x)))))) 
    return denorm

def destandardize_data(data, mean, std):
    return data * std + mean

if __name__ == '__main__':

    input_df = generate_input_data()
    print(input_df)
    mean, std, max, min = get_values_from_pkl('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/values.pkl')
    
    print('Normalization and standardization of the input data...')
    norm_df = normalize_data(input_df, max, min)
    norm_df = norm_df.drop(columns=['Mod index'])
    norm_df = norm_df[['Wo', 'Vds', 'Temp', 'Nd']] # for some reason, normalization inverts the order of the columns...
    print(norm_df)
    # std_df = standardize_data(input_df, mean, std)
    # std_df = std_df.drop(columns=['Mod index'])

    print('Predicting the data...')
    norm_prediction = prediction(norm_df, norm_model_path)
    # std_prediction = prediction(std_df, std_model_path)
    # fine_prediction = prediction(norm_df, fine_model_path)

    # Append 'Mod index' as the last column
    norm_df['Mod index'] = norm_prediction.flatten()
    # std_df['Mod index'] = std_prediction.flatten()
    # norm_df['Mod index'] = fine_prediction.flatten()

    norm_df = norm_df[['Wo', 'Vds', 'Temp', 'Nd', 'Mod index']]
    # std_df = std_df[['Wo', 'Vds', 'Temp', 'Nd', 'Mod index']]

    print('Denormalization and destandardization of the prediction data...')
    denorm_df = denormalize_data(norm_df, max, min)
    # destd_df = destandardize_data(std_df, mean, std)
    # denorm_df['Wo'] = 200.0

    print(denorm_df)
    # print(destd_df)
    
    print('Saving the prediction data...')
    denorm_df.to_csv(norm_prediction_path, index=False)
    # destd_df.to_csv(std_prediction_path, index=False)
    # denorm_df.to_csv(fine_prediction_path, index=False)

    print('Prediction data saved successfully.')