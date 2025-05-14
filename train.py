#!/usr/bin/env python3

import argparse
import datetime
import os
from typing import Optional

import tensorflow as tf

import data
import nn_model

def main(model_file: Optional[str], epochs: int, log_dir: Optional[str] = None, augmentation: data.AugmentMode = data.AugmentMode.OFF):
    model = nn_model.create_model()

    model.summary()

    #(train_images, train_labels), (test_images, test_labels) = data.make_traintest_sets()

    dataprepper = data.PrepDataset(
        batch_size=64, 
        aggressive_augmentation=(augmentation == data.AugmentMode.AGGRESSIVE),
        rotation_factor=0.10, 
        zoom_factor=0.10, 
        translation_factor=0.10
    )
                    
    # Load CIFAR-10 dataset
    dataprepper.load_data(tf.keras.datasets.cifar10.load_data())
    
    # Create training dataset with augmentation
    train_ds = dataprepper.create_dataset(dataprepper.train_images, dataprepper.train_labels, augment=(augmentation != data.AugmentMode.OFF))
    
    # Create test dataset without augmentation
    test_ds = dataprepper.create_dataset(dataprepper.test_images, dataprepper.test_labels, augment=False)
    

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    if log_dir is None:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # history = model.fit(train_images, train_labels, epochs=10, 
    #                     validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])
    history = model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=test_ds, 
        callbacks=[tensorboard_callback]
    )

    #test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    if model_file is not None:
        print(f"Saving model to {model_file}")
        model.save(model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image recognition CNN")
    parser.add_argument("output_file", nargs='?', type=str, default=None, help="Path to the checkpoint file that will be created")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for TensorBoard logs (default: logs/fit/YYYmmdd-HHMMSS).")
    parser.add_argument("--augmentations", default=data.AugmentMode.OFF, choices=[data.AugmentMode.OFF, data.AugmentMode.BASIC, data.AugmentMode.AGGRESSIVE])
    
    args = parser.parse_args()
    print(f"Training epochs: {args.epochs}")

    main(args.output_file, args.epochs, log_dir=args.log_dir, augmentation=data.AugmentMode(args.augmentations))
