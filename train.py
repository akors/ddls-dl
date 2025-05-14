#!/usr/bin/env python3

import argparse
import datetime
import os
from typing import Optional

import tensorflow as tf

import data
import nn_model

def main(model_file: Optional[str], epochs: int, log_dir: Optional[str] = None):
    model = nn_model.create_model()

    model.summary()

    (train_images, train_labels), (test_images, test_labels) = data.make_traintest_sets()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    if log_dir is None:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

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
    
    args = parser.parse_args()
    print(f"Training epochs: {args.epochs}")

    main(args.output_file, args.epochs, log_dir=args.log_dir)
