import pandas as pd
import numpy as np
import pickle as pkl
import itertools
import keras

# ========== CONSTANTS ==========

norm_model_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model_norm.keras'
fine_model_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/simpler_model.keras'

values_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/values.pkl'
fine_values_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/fine_values.pkl'

norm_prediction_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/norm_prediction.csv'
fine_prediction_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/fine_prediction.csv'

datamode = 'reduced' # Modes: 'exit', 'reduced'

# ========== FUNCTIONS ==========

def generate_input_data():

    # Define the input data

    if datamode == 'exit':

        Wo = np.arange(200, 356, 4) # number of points: 40
        Vds = np.arange(1, 61, 1) # number of points: 60
        Temp = np.array([300, 400, 500]) # number of points: 3
        Nd = np.arange(0.5, 10.5, 0.5)*1e24 # number of points: 20

        #Create the input dataframe as an iteration of all the possible combinations of the input data
        combinations = itertools.product(Wo, Vds, Temp, Nd)
        input_df = pd.DataFrame(combinations, columns=['Wo', 'Vds', 'Temp', 'Nd'])
    
    elif datamode == 'reduced':

        Vds = np.arange(1, 51, 1) # number of points: 50
        Temp = np.array([300, 400, 500]) # number of points: 3
        Nd = np.arange(0.5, 10.5, 0.5)*1e24 # number of points: 20

        combinations = itertools.product(Vds, Temp, Nd)
        input_df = pd.DataFrame(combinations, columns=['Vds', 'Temp', 'Nd'])
    
    else:
        raise ValueError('Invalid data mode creating the input data.')

    return input_df

def get_values_from_pkl(values_path):
    if datamode == 'exit':
        with open(values_path, 'rb') as f:
            values = pkl.load(f)
            max = values['max']
            min = values['min']
    if datamode == 'reduced':
        with open(fine_values_path, 'rb') as f:
            values = pkl.load(f)
            max = values['max']
            min = values['min']

    return max, min

def normalize_data(data, max, min):
    data = (data - min) / (max - min)
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

if __name__ == '__main__':

    input_df = generate_input_data()
    print(input_df)
    max, min = get_values_from_pkl(values_path) if datamode == 'exit' else get_values_from_pkl(fine_values_path)
    
    print('Normalization and standardization of the input data...')
    norm_df = normalize_data(input_df, max, min)
    norm_df = norm_df.drop(columns=['Mod index'])

    if datamode == 'exit': 
        norm_df = norm_df[['Wo', 'Vds', 'Temp', 'Nd']] # for some reason, normalization inverts the order of the columns...
    elif datamode == 'reduced':
        norm_df = norm_df[['Vds', 'Temp', 'Nd']]
    else:
        raise ValueError('Invalid data mode normalizing the input data.')
    print(norm_df)

    print('Predicting the data...')
    if datamode == 'exit':
        norm_prediction = prediction(norm_df, norm_model_path)
    elif datamode == 'reduced':
        norm_prediction = prediction(norm_df, fine_model_path)
    else:
        raise ValueError('Invalid data mode predicting the data.')

    # Append 'Mod index' as the last column
    norm_df['Mod index'] = norm_prediction.flatten()

    if datamode == 'exit':
        norm_df = norm_df[['Wo', 'Vds', 'Temp', 'Nd', 'Mod index']]
    elif datamode == 'reduced':
        norm_df = norm_df[['Vds', 'Temp', 'Nd', 'Mod index']]
    else:
        raise ValueError('Invalid data mode appending the prediction data.')

    print('Denormalization and destandardization of the prediction data...')
    denorm_df = denormalize_data(norm_df, max, min)

    if datamode == 'reduced':
        denorm_df['Wo'] = 200.0
        denorm_df = denorm_df[['Wo', 'Vds', 'Temp', 'Nd', 'Mod index']]


    print(denorm_df)
    # print(destd_df)
    
    print('Saving the prediction data...')

    if datamode == 'exit':
        denorm_df.to_csv(norm_prediction_path, index=False)
    elif datamode == 'reduced':
        denorm_df.to_csv(fine_prediction_path, index=False)
    else:
        raise ValueError('Invalid data mode saving the prediction data.')

    print('Prediction data saved successfully.')