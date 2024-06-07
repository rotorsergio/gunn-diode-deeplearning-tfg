import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

norm_dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/normalized.csv'
std_dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/standardized.csv'

print(tf.__version__)
print(keras.__version__)

# =================== NEURAL NETWORK ====================

def create_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=32)
Z_train, Z_val, w_train, w_val = train_test_split(Z, w, test_size=0.2, random_state=32)

# Create the first model for normalized data
model_norm = create_model()
model_norm.compile(optimizer='adam', loss='mean_squared_error')

# Train the first model
print('Training the model for normalized data...')
history_norm = model_norm.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose='1')

# Save the first model
model_norm.save('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/model_norm.h5')

# Create the second model for standardized data
model_std = create_model()  # Assuming create_model() is a function that returns a new instance of your model
model_std.compile(optimizer='adam', loss='mean_squared_error')

# Train the second model
print('Training the model for standardized data...')
history_std = model_std.fit(Z_train, w_train, epochs=100, validation_data=(Z_val, w_val), verbose='1')

# Save the second model
model_std.save('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/model_std.h5')

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