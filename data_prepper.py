import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from augmentation import DataAugmentation, AggressiveDataAugmentation

class PrepDataset:
    def __init__(   self, 
                    batch_size=64, 
                    aggressive_augmentation=False,
                    rotation_factor=0.10, 
                    zoom_factor=0.10, 
                    translation_factor=0.10):
        self.batch_size = batch_size
        self.model = None
        self.history = None
        if aggressive_augmentation:
            self.augmenter = AggressiveDataAugmentation(rotation_factor, zoom_factor, translation_factor)
        else:
            self.augmenter = DataAugmentation(rotation_factor, zoom_factor, translation_factor)
        
    def load_data(self, dataset):
        """Load and normalize CIFAR-10 dataset"""
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = dataset
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
    
"""
    def train(self, train_ds, test_ds, epochs=100):
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
    prepper = PrepDataset(
                    batch_size=64, 
                    aggressive_augmentation=False,
                    rotation_factor=0.10, 
                    zoom_factor=0.10, 
                    translation_factor=0.10
                    )
                    
    prepper.load_data(datasets.cifar10.load_data())
    train_ds = prepper.create_dataset(prepper.train_images, prepper.train_labels, augment=True)
    test_ds = prepper.create_dataset(prepper.test_images, prepper.test_labels, augment=False)


"""