import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from scipy.stats import pearsonr
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/200nm.csv'
model_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/simpler.keras'
lossplot_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/loss_simpler.png'
dispersionplot_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/dispersion_simpler.png'

print('SIMPLER ANN MODEL. SCALED DATA.\n')
print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

# ========== VARIABLES ==========

# activation_functions = ['relu', 'tanh', 'sigmoid', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential']
# optimizers = ['adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']

densidad: int = 20
funcion_activacion = 'relu'
funcion_perdida = 'huber'
optimizador = 'adam'
random_seed = 47
number_epochs: int = 200

# =================== NEURAL NETWORK ====================

model = keras.Sequential(
    [
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(densidad, activation=funcion_activacion),
        keras.layers.Dense(densidad, activation=funcion_activacion),
        keras.layers.Dense(densidad, activation=funcion_activacion),
        keras.layers.Dense(1, activation='linear')
    ]
)

model.summary()
model.compile(optimizer=optimizador, loss=funcion_perdida)

# =================== PREPARE DATA ====================

data = pd.read_csv(dataset_path).drop(columns='Wo')

max = data.max()
min = data.min()
scaled = (data - min) / (max - min)
scaled_array = np.array(scaled)

# Last column: target of prediction. The rest: input data
X = scaled.iloc[:, :-1]
y = scaled.iloc[:, -1]

Xarray = scaled_array[:, :-1]
yarray = scaled_array[:, -1]

# Split the dataset into 80% training data and 20% validation data
# Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=random_seed)
Xtrain, Xval, ytrain, yval = train_test_split(Xarray, yarray, test_size=0.2, random_state=random_seed)
'''
Xtrain = Xtrain.reset_index(drop=True)
Xval = Xval.reset_index(drop=True)
ytrain = ytrain.reset_index(drop=True)
yval = yval.reset_index(drop=True)
'''

# =================== TRAINING ====================

print('Training the model for normalized data...')
history = model.fit(Xtrain, ytrain, epochs=number_epochs, validation_data=(Xval, yval), batch_size=32) # Default batch_size = 32 SEEMS TO MATTER A LOT! DON'T USE LOW VALUES
model.save(model_path)

# Print the training history from the model

plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for normalized data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(lossplot_path)
plt.close()

# Represent prediction versus real data

ytrain_predicted = model.predict(Xtrain)
yval_predicted = model.predict(Xval)

ytrain_predicted = ytrain_predicted.flatten()
yval_predicted = yval_predicted.flatten()

ytrain = ytrain.flatten()
yval = yval.flatten()

ytrain = pd.DataFrame(ytrain, columns=['Real train'])
yval = pd.DataFrame(yval, columns=['Real validation'])
predicted_train = pd.DataFrame(ytrain_predicted, columns=['Predicted train'])
predicted_val = pd.DataFrame(yval_predicted, columns=['Predicted validation'])

train_compare = pd.concat([ytrain, predicted_train], axis=1)
val_compare = pd.concat([yval, predicted_val], axis=1)
print(train_compare)
print(val_compare)

plt.figure(figsize=(16,9))
plt.subplot(1, 2, 1)
plt.plot(ytrain, ytrain_predicted, 'o', label='Train', scalex=False, scaley=False)
plt.title('Train vs prediction')
plt.subplot(1, 2, 2)
plt.plot(yval, yval_predicted, 'o', label='Validation', scalex=False, scaley=False)
plt.title('Validation vs prediction')