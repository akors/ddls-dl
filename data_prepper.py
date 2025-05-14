import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from augmentation import DataAugmentation, AggressiveDataAugmentation

class PrepDataset:
    def __init__(self, batch_size=64, aggressive_augmentation=False):
        self.batch_size = batch_size
        self.model = None
        self.history = None
        if aggressive_augmentation:
            self.augmenter = AggressiveDataAugmentation()
        else:
            self.augmenter = DataAugmentation()
        
    def load_data(self):
        """Load and normalize CIFAR-10 dataset"""
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
        return (self.train_images, self.train_labels), (self.test_images, self.test_labels)
    
    def create_dataset(self, images, labels, augment=False):
        """Create data pipeline"""
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.shuffle(10000)
        
        if augment:
            ds = ds.map(lambda x, y: (self.augmenter.augment(tf.cast(x, tf.float32) / 255.0), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
            
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds