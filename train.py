#!/usr/bin/env python3

import argparse
import datetime
import os
from typing import Optional

import tensorflow as tf

import data
import nn_model

def main(model_file: Optional[str], epochs: int, batchsize: int = 256, log_dir: Optional[str] = None, augmentation: data.AugmentMode = data.AugmentMode.OFF):
    model = nn_model.create_model()

    model.summary()

    dataprepper = data.PrepDataset(
        batch_size=batchsize, 
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

    history = model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=test_ds, 
        callbacks=[tensorboard_callback]
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    if model_file is not None:
        print(f"Saving model to {model_file}")
        model.save(model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image recognition CNN")
    parser.add_argument("output_file", nargs='?', type=str, default=None, help="Path to the checkpoint file that will be created")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for TensorBoard logs (default: logs/fit/YYYmmdd-HHMMSS).")
    parser.add_argument("--augmentations", default="off", choices=["off", "basic", "aggressive"])
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (default: 256).")

    args = parser.parse_args()

    if args.augmentations == "off":
        augmentation = data.AugmentMode.OFF
    elif args.augmentations == "basic":
        augmentation = data.AugmentMode.BASIC
    elif args.augmentations == "aggressive":
        augmentation = data.AugmentMode.AGGRESSIVE
    else:
        raise KeyError(f"Unknown augmentation mode {args.augmentations}")


    print(f"Training epochs: {args.epochs}")

    main(args.output_file, args.epochs, batchsize=args.batch_size, log_dir=args.log_dir, augmentation=augmentation)
