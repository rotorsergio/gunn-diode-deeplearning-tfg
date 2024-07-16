import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable Tensorflow warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import keras
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

norm_dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/normalized.csv'
reduced_norm_dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/norm_200nm.csv'

model_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model_norm.keras'
reduced_model_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/simpler_model.keras'

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

# ========== VARIABLES ==========

# activation_functions = ['relu', 'tanh', 'sigmoid', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential']
# optimizers = ['adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']

random_seed: int = 1
activation_function = 'sigmoid'
optimizer_choose = 'adam'
loss_function = 'mean_squared_error'
density: int = 10
number_epochs: int = 2000
datamode = 'reduced' # Modes: 'exit', 'reduced'
train_model: bool = True

# =================== NEURAL NETWORK ====================

def create_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=((4,) if datamode == 'exit' else (3,))),
            keras.layers.Dense(density, activation=activation_function),
            keras.layers.Dense(density, activation=activation_function),
            keras.layers.Dense(1, activation='linear')
        ]
    )

    model.summary() # print the model summary

    return model

# =================== TRAINING ====================

norm_df = pd.read_csv(norm_dataset_path) if datamode == 'exit' else pd.read_csv(reduced_norm_dataset_path)

# Last column: target of prediction. The rest: input data
X = norm_df.iloc[:, :-1]
y = norm_df.iloc[:, -1]

# Split the dataset into 80% training data and 20% validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

if train_model == True:

    # Create the model for normalized data
    model_norm = create_model()
    model_norm.compile(optimizer=optimizer_choose, loss=loss_function)

    print('Training the model for normalized data...')
    history_norm = model_norm.fit(X_train, y_train, epochs=number_epochs, validation_data=(X_val, y_val), batch_size=32)
    if datamode == 'exit':
        model_norm.save(model_path, overwrite=True)
    elif datamode == 'reduced':
        model_norm.save(reduced_model_path, overwrite=True)
    else:
        raise ValueError('Invalid data mode while training the model.')

    # Print the training history from both models

    plt.figure(figsize=(8, 6))
    plt.plot(history_norm.history['loss'])
    plt.plot(history_norm.history['val_loss'])
    plt.title('Model loss for normalized data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    if datamode == 'exit':
        plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/loss_function_norm.png')
    elif datamode == 'reduced':
        plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/loss_function_simpler.png')
    else:
        raise ValueError('Invalid data mode while saving the loss function plot.')

else:
    if datamode == 'exit':
        model_norm = keras.models.load_model(model_path)
    elif datamode == 'reduced':
        model_norm = keras.models.load_model(reduced_model_path)
    else:
        raise ValueError('Invalid data mode while loading the model.')


# Represent prediction versus real data

y_pred_norm_train = model_norm.predict(X_train) # type: ignore --> avoid interpreter false error in model.predict
y_pred_norm_val = model_norm.predict(X_val) # type: ignore

r_train_norm = pearsonr(y_train.values, y_pred_norm_train.flatten())[0]
r_val_norm = pearsonr(y_val.values, y_pred_norm_val.flatten())[0]

plt.figure(figsize=(8,6))

plt.subplot(1, 2, 1)
plt.plot(y_train, y_pred_norm_train, 'o', label='Train')
plt.xlabel('Real training data')
plt.ylabel('Predicted training data')
plt.plot([0,1],[0,1], 'r') # Diagonal line
plt.text(0.1, 0.9, f'R = {r_train_norm:.4f}', fontsize=12, color='red')

plt.subplot(1, 2, 2)
plt.plot(y_val, y_pred_norm_val, 'o', label='Validation')
plt.plot([0,1],[0,1], 'r') # Diagonal line
plt.text(0.1, 0.9, f'R = {r_val_norm:.4f}', fontsize=12, color='red')
plt.xlabel('Real validation data')
plt.ylabel('Predicted validation data')

if datamode == 'exit':
    plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/dispersion_norm.png')
elif datamode == 'reduced':
    plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/dispersion_simpler.png')
else:
    raise ValueError('Invalid data mode while saving the dispersion plot.')

plt.close()