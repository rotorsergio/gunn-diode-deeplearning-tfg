import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

norm_dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/normalized.csv'
std_dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/standardized.csv'

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

# ========== VARIABLES ==========

# activation_functions = ['relu', 'tanh', 'sigmoid', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential']
# optimizers = ['adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']

random_seed = 1
activation_function = 'relu'
optimizer_choose = 'adam'
loss_function = 'mean_absolute_error'
density: int = 20
number_epochs: int = 200

# =================== NEURAL NETWORK ====================

def create_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(density, activation=activation_function),
            keras.layers.Dense(density, activation=activation_function),
            keras.layers.Dense(density, activation=activation_function),
            keras.layers.Dense(1, activation='linear')
        ]
    )

    model.summary() # print the model summary

    return model

# =================== TRAINING ====================

norm_df = pd.read_csv(norm_dataset_path)
std_df = pd.read_csv(std_dataset_path)

# Last column: target of prediction. The rest: input data
X = norm_df.iloc[:, :-1]
y = norm_df.iloc[:, -1]
Z = std_df.iloc[:, :-1]
w = std_df.iloc[:, -1]

# Split the dataset into 80% training data and 20% validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)
Z_train, Z_val, w_train, w_val = train_test_split(Z, w, test_size=0.2, random_state=random_seed)

# Create the first model for normalized data
model_norm = create_model()
model_norm.compile(optimizer=optimizer_choose, loss=loss_function)

print('Training the model for normalized data...')
history_norm = model_norm.fit(X_train, y_train, epochs=number_epochs, validation_data=(X_val, y_val), batch_size=32)
model_norm.save('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model_norm.keras')

# Create the second model for standardized data
model_std = create_model()  # Assuming create_model() is a function that returns a new instance of your model
model_std.compile(optimizer=optimizer_choose, loss=loss_function)

print('Training the model for standardized data...')
history_std = model_std.fit(Z_train, w_train, epochs=number_epochs, validation_data=(Z_val, w_val), batch_size=8)
model_std.save('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg//models/model_std.keras')

# Print the training history from both models

plt.figure(figsize=(16, 9))
plt.plot(history_norm.history['loss'])
plt.plot(history_norm.history['val_loss'])
plt.title('Model loss for normalized data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/loss_function_norm.png')

plt.figure(figsize=(16, 9))
plt.plot(history_std.history['loss'])
plt.plot(history_std.history['val_loss'])
plt.title('Model loss for standardized data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/loss_function_std.png')

# Represent prediction versus real data

y_pred_norm_train = model_norm.predict(X_train)
y_pred_norm_val = model_norm.predict(X_val)

plt.figure(figsize=(16, 9))
plt.subplot(1, 2, 1)
plt.plot(y_train, y_pred_norm_train, 'o', label='Train')
plt.subplot(1, 2, 2)
plt.plot(y_val, y_pred_norm_val, 'o', label='Validation')
plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/dispersion_norm.png')
plt.close()

w_pred_std_train = model_std.predict(Z_train)
w_pred_std_val = model_std.predict(Z_val)

plt.figure(figsize=(16, 9))
plt.subplot(1, 2, 1)
plt.plot(w_train, w_pred_std_train, 'o')
plt.title('Training versus predicted data')
plt.subplot(1, 2, 2)
plt.plot(w_val, w_pred_std_val, 'o')
plt.title('Validation versus predicted data')
plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/dispersion_std.png')
plt.close()