
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

def create_model():
    input = keras.Input(shape=(32, 32, 3), name="img")


    # %% convolutional layers
    x = layers.Conv2D(8, (3, 3), input_shape=(32, 32, 3))(input)
    x = layers.ReLU()(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # %% dense final layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(10)(x)
    output = layers.Softmax()(output)

    model = keras.Model(input, output, name="cnn")

    return model
