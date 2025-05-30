
from enum import Enum
from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
# from scipy.ndimage import zoom

import augmentation
import pandas as pd
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# class syntax
class AugmentMode(Enum):
    OFF = "off"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"


class PrepDataset:
    """
    A class for preparing and managing datasets for deep learning models.
    
    This class handles loading data, creating TensorFlow datasets with proper
    batching and prefetching, applying data augmentation, and training models.
    """
    def __init__(   self, 
                    batch_size=64, 
                    aggressive_augmentation=False,
                    rotation_factor=0.10, 
                    zoom_factor=0.10, 
                    translation_factor=0.10):
        """
        Initialize the dataset preparation pipeline.
        
        Args:
            batch_size: Number of samples per batch
            aggressive_augmentation: Whether to use more aggressive augmentation techniques
            rotation_factor: Maximum rotation angle as a fraction of 2π
            zoom_factor: Maximum zoom factor
            translation_factor: Maximum translation as fraction of image dimensions
        """
        self.batch_size = batch_size
        self.model = None
        self.history = None
        if aggressive_augmentation:
            self.augmenter = augmentation.AggressiveDataAugmentation(rotation_factor, zoom_factor, translation_factor)
        else:
            self.augmenter = augmentation.DataAugmentation(rotation_factor, zoom_factor, translation_factor)
        
    def load_data(self, dataset):
        """
        Load and normalize CIFAR-10 dataset
        
        Args:
            dataset: A tuple of (train_data, test_data) from tf.keras.datasets
            
        Returns:
            Tuple of (train_images, train_labels), (test_images, test_labels)
        """
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = dataset
        
         # Combine train and test sets
        all_images = np.concatenate([self.train_images, self.test_images], axis=0)
        all_labels = np.concatenate([self.train_labels, self.test_labels], axis=0)
        # Load indices from CSV files
        train_indices = pd.read_csv('y_train.csv').iloc[:, 0].values
        val_indices = pd.read_csv('y_val.csv').iloc[:, 0].values
        
        # Split data according to indices
        self.train_images = all_images[train_indices]
        self.train_labels = all_labels[train_indices]
        self.test_images = all_images[val_indices] 
        self.test_labels = all_labels[val_indices]
        print(self.train_images.shape)
        print(self.train_labels.shape)
        print(self.test_images.shape)
        print(self.test_labels.shape)
        
        
        
        # One-hot encode the labels
        self.train_labels = tf.keras.utils.to_categorical(self.train_labels, num_classes=10)
        self.test_labels = tf.keras.utils.to_categorical(self.test_labels, num_classes=10)
        
        return (self.train_images, self.train_labels), (self.test_images, self.test_labels)
    
    def create_dataset(self, images, labels, augment=False):
        """
        Create data pipeline with TensorFlow Dataset API
        
        This method:
        1. Creates a dataset from image and label tensors
        2. Shuffles the dataset (at the epoch level with buffer size 10000)
        3. Applies normalization and optional augmentation
        4. Batches and prefetches data for optimal performance
        
        Args:
            images: Image data as numpy array
            labels: Label data as numpy array
            augment: Whether to apply data augmentation
            
        Returns:
            A tf.data.Dataset object ready for training
        """
        # Create dataset from tensors
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        # Shuffle at epoch level with buffer size 10000
        ds = ds.shuffle(60000)
        
        if augment:
            # Apply augmentation at individual image level during iteration
            ds = ds.map(lambda x, y: (self.augmenter.augment(tf.cast(x, tf.float32) / 255.0), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            # Just normalize without augmentation
            ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
            
        # Batch and prefetch for performance
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


def make_traintest_sets() -> Tuple[Tuple[ArrayLike, ArrayLike], Tuple[ArrayLike]]:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)
