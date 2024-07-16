import pandas as pd
import numpy as np
import pickle as pkl
import keras

# ========== CONSTANTS ==========
desired_number_of_points: int = 21

norm_dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/normalized.csv'
norm_model_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model_norm.keras'
norm_prediction_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/norm_prediction.csv'

predicted_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/og_prediction.csv'

# ========== FUNCTIONS ==========

def get_values_from_pkl(values_path):
    with open(values_path, 'rb') as f:
        values = pkl.load(f)
        mean = values['mean']
        std = values['std']
        max = values['max']
        min = values['min']
        print('Pickle saved values:\n', values)
    return mean, std, max, min

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
    return data * (max - min) + min

# ========== M A I N ==========

complete_df = pd.read_csv(norm_dataset_path)
input_df = complete_df.drop(columns=['Mod index'])
print(input_df)
print('Original shape: ', input_df.shape)
mean, std, max, min = get_values_from_pkl('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/values.pkl')

print('Predicting the data...')
norm_prediction = prediction(input_df, norm_model_path)
predicted_values = norm_prediction.flatten()
real_values = complete_df['Mod index'].values
normalized_error = real_values - predicted_values
print('Error of the prediction: \n', normalized_error)

# Create dataset with the predicted values

predicted_modindex = pd.DataFrame(predicted_values, columns=['Mod index'])
predicted_df = pd.concat([input_df, predicted_modindex], axis=1)
predicted_df = predicted_df*(max-min)+min
print(predicted_df)

# Export predicted_df to csv to plot
predicted_df.to_csv(predicted_path, index=False)