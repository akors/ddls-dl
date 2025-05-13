
import tensorflow.keras

from typing import Tuple
from numpy.typing import ArrayLike

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def make_traintest_sets() -> Tuple[Tuple[ArrayLike, ArrayLike], Tuple[ArrayLike]]:
    (train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)
