import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

print(tf.__version__)
print(keras.__version__)

# =================== NEURAL NETWORK ====================
# Sequential model
# model = tf.python.keras.Sequential() # i dont know xd

model = keras.Sequential(
    [
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1)
    ]
)

model.summary() # print the model summary

# =================== TRAINING ====================

# The model needs to be compiled before training
# The optimizer is the algorithm used to update the weights of the model
# The loss function is the function that the model tries to minimize
# The metrics are the metrics used to evaluate the model
# The most common loss function for regression problems is the mean squared error
# The most common optimizer is the Adam optimizer
# The most common metric for regression problems is the mean squared error

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Now the model is ready to be trained with model.fit()

dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/normalized.csv'
df = pd.read_csv(dataset_path)

# Last column: target of prediction. The rest: input data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into 80% training data and 20% validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=32)

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose='1')

# Save the model
model.save('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/model.h5')

# Print the training history
plt.figure(figsize=(16, 9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/loss_function.png')