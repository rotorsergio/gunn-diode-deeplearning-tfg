import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
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

random_seed: int = 1
activation_function = 'sigmoid'
optimizer_choose = 'adam'
loss_function = 'mean_squared_error'
density: int = 10
number_epochs: int = 800
train_model: bool = True

# =================== NEURAL NETWORK ====================

def create_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(density, activation=activation_function),
            keras.layers.Dense(density, activation=activation_function),
            keras.layers.Dense(1, activation='linear')
        ]
    )

    model.summary() # print the model summary

    return model

# =================== TRAINING ====================

norm_df = pd.read_csv(norm_dataset_path)

# Last column: target of prediction. The rest: input data
X = norm_df.iloc[:, :-1]
y = norm_df.iloc[:, -1]

# Split the dataset into 80% training data and 20% validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

if train_model == True:

    # Create the first model for normalized data
    model_norm = create_model()
    model_norm.compile(optimizer=optimizer_choose, loss=loss_function)

    print('Training the model for normalized data...')
    history_norm = model_norm.fit(X_train, y_train, epochs=number_epochs, validation_data=(X_val, y_val), batch_size=32)
    model_norm.save('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model_norm.keras', overwrite=True)

    # Print the training history from both models

    plt.figure(figsize=(16, 9))
    plt.plot(history_norm.history['loss'])
    plt.plot(history_norm.history['val_loss'])
    plt.title('Model loss for normalized data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/loss_function_norm.png')

else:
    model_norm = keras.models.load_model('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model_norm.keras')


# Represent prediction versus real data

y_pred_norm_train = model_norm.predict(X_train)
y_pred_norm_val = model_norm.predict(X_val)

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
plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/dispersion_norm.png')
plt.close()