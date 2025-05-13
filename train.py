#!/usr/bin/env python3

import argparse
from typing import Optional

import tensorflow as tf

import data
import nn_model

def main(model_file: Optional[str], epochs: int):
    model = nn_model.create_model()

    model.summary()

    (train_images, train_labels), (test_images, test_labels) = data.make_traintest_sets()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

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
    
    args = parser.parse_args()
    print(f"Training epochs: {args.epochs}")

    main(args.output_file, args.epochs)
