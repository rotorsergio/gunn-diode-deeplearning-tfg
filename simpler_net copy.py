import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
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

random_seed = 69
number_epochs: int = 400
train_model: bool = True

# =================== NEURAL NETWORK ====================

model = keras.Sequential(
    [
        keras.layers.Input(shape=(3,), name='input_layer'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1, name='output_layer')
    ]
)

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')

# =================== PREPARE DATA ====================

data = pd.read_csv(dataset_path)
# data = data.drop(columns=['Intervalley', 'FMA'])
data = data.drop(columns='Wo')
scaler = MinMaxScaler() # Define scaler. Options: MinMaxScaler(), StandardScaler(), RobustScaler(), Normalizer()
df = pd.DataFrame(scaler.fit_transform(data), columns=data.columns) # Normalize data. fit() computes the min and max values of the data, and transform() applies the transformation to the data.

# Last column: target of prediction. The rest: input data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the dataset into 80% training data and 20% validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# =================== TRAINING ====================

print('Training the model for normalized data...')
history = model.fit(X_train, y_train, epochs=number_epochs, validation_data=(X_val, y_val), batch_size=64) # Defalut batch_size = 32 SEEMS TO MATTER A LOT! DON'T USE LOW VALUES
model.save(model_path)

# Print the training history from both models

plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for normalized data')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(lossplot_path)

# Represent prediction versus real data

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

plt.figure(figsize=(16,9))
plt.subplot(1, 2, 1)
plt.plot(y_train, y_pred_train, 'o', label='Train', scalex=False, scaley=False)
plt.subplot(1, 2, 2)
plt.plot(y_val, y_pred_val, 'o', label='Validation', scalex=False, scaley=False)
plt.show()
plt.savefig(dispersionplot_path)
plt.close()

# =================== PREDICTION ==================== DONT USE NOW
# model = keras.models.load_model(model_path)

# Generate a set of new X_pred data to predict using iterator


# X_pred = pd.DataFrame(scaler.transform(X_pred), columns=X_pred.columns)



# y_pred = model.predict(X_pred)