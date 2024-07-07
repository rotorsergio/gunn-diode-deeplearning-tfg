import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataset_path = 'C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/datasets/exit.csv'

random_seed = 9
activation_function = 'sigmoid'
optimizer_choose = 'adam'
loss_function = 'mean_squared_error'
density: int = 10
number_epochs: int = 400

def create_model():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(density, activation=activation_function),
            keras.layers.Dense(1, activation=activation_function)
        ]
    )

    model.summary()

    return model

df = pd.read_csv(dataset_path)
df = df.drop(columns=['Intervalley', 'FMA'])
df['Nd']=df['Nd']/1e24
df['Temp']=df['Temp']/100
print(df.describe())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

model = create_model()
model.compile(optimizer=optimizer_choose, loss=loss_function, metrics=['mean_squared_error'])

history = model.fit(X_train, y_train, epochs=number_epochs, validation_data=(X_test, y_test), batch_size=8)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/plots/loss.png')
plt.show()

model.save('C:/Users/sergi/repositorios/gunn-diode-deeplearning-tfg/models/model.keras')