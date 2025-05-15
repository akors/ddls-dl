from tensorflow import keras
from tensorflow.keras import layers

def basic_block(x, in_channels, out_channels, downsample):
    stride = 2 if downsample else 1
    shortcut = x

    # First conv → BN → ReLU
    x = layers.Conv2D(out_channels, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second conv → BN
    x = layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Downsample the shortcut if needed
    if downsample or in_channels != out_channels:
        shortcut = layers.Conv2D(out_channels, kernel_size=1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def create_model(use_softmax=True):
    inputs = keras.Input(shape=(32, 32, 3), name="img")

    # Initial Conv-BN-ReLU block (same as nn.Sequential block in PyTorch)
    x = layers.Conv2D(24, kernel_size=7, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # BasicBlock(24 → 48 with downsample)
    x = basic_block(x, in_channels=24, out_channels=48, downsample=True)
    # BasicBlock(48 → 48)
    x = basic_block(x, in_channels=48, out_channels=48, downsample=False)
    # BasicBlock(48 → 48)
    x = basic_block(x, in_channels=48, out_channels=48, downsample=False)

    # Global average pooling + final dense
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10)(x)

    if use_softmax:
        x = layers.Softmax()(x)

    return keras.Model(inputs=inputs, outputs=x, name="ThreeBlockResNet")
