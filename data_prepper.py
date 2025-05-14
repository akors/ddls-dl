import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from augmentation import DataAugmentation, AggressiveDataAugmentation

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
            self.augmenter = AggressiveDataAugmentation(rotation_factor, zoom_factor, translation_factor)
        else:
            self.augmenter = DataAugmentation(rotation_factor, zoom_factor, translation_factor)
        
    def load_data(self, dataset):
        """
        Load and normalize CIFAR-10 dataset
        
        Args:
            dataset: A tuple of (train_data, test_data) from tf.keras.datasets
            
        Returns:
            Tuple of (train_images, train_labels), (test_images, test_labels)
        """
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = dataset
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
        ds = ds.shuffle(10000)
        
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
    
"""
def train(self, train_ds, test_ds, epochs=100):
    '''
    Train the model with early stopping
    
    Args:
        train_ds: Training dataset from create_dataset()
        test_ds: Validation dataset from create_dataset()
        epochs: Maximum number of epochs to train
        
    Returns:
        Training history object
    '''
    # Train model with early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    self.model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    self.history = self.model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=test_ds, 
        callbacks=[early_stop]
    )
    
    return self.history


if __name__ == '__main__':
    # Example usage of the PrepDataset class
    prepper = PrepDataset(
                    batch_size=64, 
                    aggressive_augmentation=False,
                    rotation_factor=0.10, 
                    zoom_factor=0.10, 
                    translation_factor=0.10
                    )
                    
    # Load CIFAR-10 dataset
    prepper.load_data(datasets.cifar10.load_data())
    
    # Create training dataset with augmentation
    train_ds = prepper.create_dataset(prepper.train_images, prepper.train_labels, augment=True)
    
    # Create test dataset without augmentation
    test_ds = prepper.create_dataset(prepper.test_images, prepper.test_labels, augment=False)
    
    # Train the model
    history = prepper.train(train_ds, test_ds)


"""