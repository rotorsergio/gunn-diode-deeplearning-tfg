import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

norm_dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/normalized.csv'
std_dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/standardized.csv'

print('Tensorflow version: ', tf.__version__)
print('Keras version: ', keras.__version__)

# ========== VARIABLES ==========

# activation_functions = ['relu', 'tanh', 'sigmoid', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential']
# optimizers = ['adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']

random_seed = 47
activation_function = 'sigmoid'
optimizer_choose = 'adam'
loss_function = 'mean_squared_error' # Representation wont be changed by this!!!!!
density: int = 10
number_epochs: int = 1000

# =================== NEURAL NETWORK ====================

def create_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(density, activation=activation_function),
#            keras.layers.Dense(density, activation=activation_function),
            keras.layers.Dense(1)
        ]
    )

    model.summary() # print the model summary

    return model

# =================== TRAINING ====================

# The model needs to be compiled before training
# The optimizer is the algorithm used to update the weights of the model
# The loss function is the function that the model tries to minimize
# The metrics are the metrics used to evaluate the model
# The most common loss function for regression problems is the mean squared error
# The most common optimizer is the Adam optimizer: based con gradient descent
# The most common metric for regression problems is the mean squared error

# Now the model is ready to be trained with model.fit()

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

# Train the first model
print('Training the model for normalized data...')
history_norm = model_norm.fit(X_train, y_train, epochs=number_epochs, validation_data=(X_val, y_val), batch_size=8 ,verbose='1')

norm_predict_test = model_norm.predict(X_val) # Predict the validation data. Optimal: use a different dataset for testing
norm_score = mean_squared_error(y_val, norm_predict_test) # Calculate the error of the model after all epochs usign val_data

# Save the first model
model_norm.save('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/model_norm.keras')

# Create the second model for standardized data
model_std = create_model()  # Assuming create_model() is a function that returns a new instance of your model
model_std.compile(optimizer=optimizer_choose, loss=loss_function)

# Train the second model
print('Training the model for standardized data...')
history_std = model_std.fit(Z_train, w_train, epochs=number_epochs, validation_data=(Z_val, w_val),batch_size=8, verbose='1')

std_predict_test = model_std.predict(Z_val) # Predict the validation data. Optimal: use a different dataset for testing
std_score = mean_squared_error(w_val, std_predict_test) # Calculate the error of the model after all epochs usign val_data

# Save the second model
model_std.save('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/model_std.keras')

# Print the training history from both models

plt.figure(figsize=(16, 9))
plt.plot(history_norm.history['loss'])
plt.plot(history_norm.history['val_loss'])
plt.title('Model loss for normalized data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/loss_function_norm.png')

plt.figure(figsize=(16, 9))
plt.plot(history_std.history['loss'])
plt.plot(history_std.history['val_loss'])
plt.title('Model loss for standardized data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/loss_function_std.png')