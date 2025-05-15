#!/usr/bin/env python3

import argparse
import datetime
import os
import subprocess
from typing import Optional

import numpy as np
import tensorflow as tf

import data
import nn_model

def get_num_params(model):
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    return trainable_params.item()


def get_git_branch():
    wd = os.path.dirname(__file__)
    try:
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=subprocess.PIPE, cwd=wd)
    except FileNotFoundError:
        # Could not find git binary
        return None

    if result.returncode != 0:
        return None
    else:
        return result.stdout.decode('ascii').strip()

def get_git_revision_short_hash():
    wd = os.path.dirname(__file__)
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE, cwd=wd)
    except FileNotFoundError:
        # Could not find git binary
        return None

    if result.returncode != 0:
        return None
    else:
        return result.stdout.decode('ascii').strip()


def report_dict_to_markdown_table(report_dict):
    report_dict = dict(report_dict)
    report_dict['test_loss'] = f"{report_dict['test_loss']:.3e}"
    report_dict['train_loss'] = f"{report_dict['train_loss']:.3e}"

    headers = list(report_dict.keys())
    values = [str(v) for v in report_dict.values()]
    separators = []

    for ki in range(len(report_dict)):
        l = max(len(headers[ki]), len(values[ki]))
        separators.append("-"*l)
        headers[ki] = headers[ki] + " "* (l - len(headers[ki]))
        values[ki] = values[ki] + " "* (l - len(values[ki]))

    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(separators) + " |"
    values = "| " + " | ".join(values) + " |"
    return "\n".join([header, separator, values])

def main(
    model_file: Optional[str],
    epochs: int,
    batchsize: int = 256,
    learning_rate: float = 1e-3,
    log_dir: Optional[str] = None,
    confusion_matrix_plotfile = None,
    augmentation: data.AugmentMode = data.AugmentMode.OFF
):
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
    if log_dir is None:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', mode='auto', factor=0.5, patience=5, min_delta=1e-6, min_lr=1e-8
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    callbacks = [
        lr_schedule,
        early_stopping,
        tensorboard_callback
    ]

    if model_file is not None:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            model_file, 
            save_best_only=True, 
            monitor='val_loss', 
            mode='min', 
            verbose=1
        )

        callbacks.append(model_checkpoint_callback)
 
    history = model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=test_ds, 
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)

    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    if model_file is not None:
        print(f"Saving model to {model_file}")
        model.save(model_file)
    
    if confusion_matrix_plotfile is not None:
        from confusion_matrix import ConfusionMatrixPlotter

        # After training your model
        confusion_plotter = ConfusionMatrixPlotter(model, test_ds)

        # Save regular confusion matrix
        confusion_plotter.plot_confusion_matrix(confusion_matrix_plotfile)

    report_dict = {
        "branch": get_git_branch(),
        "commit": get_git_revision_short_hash(),
        "num_params": get_num_params(model),
        "test_loss": test_loss,
        "train_loss": model.history.history['loss'][-1], 
        "epochs": len(model.history.history['loss']),
        "batchsize": batchsize
    }

    print(report_dict_to_markdown_table(report_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image recognition CNN")
    parser.add_argument("output_file", nargs='?', type=str, default=None, help="Path to the checkpoint file that will be created")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20).")
    parser.add_argument("--log_dir", type=str, default=None, help="Directory for TensorBoard logs (default: logs/fit/YYYmmdd-HHMMSS).")
    parser.add_argument("--augmentations", default="off", choices=["off", "basic", "aggressive"])
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (default: 256).")
    parser.add_argument("--confusion_matrix", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

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

    main(
        args.output_file,
        args.epochs,
        batchsize=args.batch_size,
        learning_rate=args.learning_rate,
        log_dir=args.log_dir,
        confusion_matrix_plotfile=args.confusion_matrix,
        augmentation=augmentation)
