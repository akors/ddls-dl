import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from data import class_names

class ConfusionMatrixPlotter:
    """
    A class for creating and saving confusion matrix plots for classification results.
    """
    def __init__(self, model, test_dataset):
        """
        Initialize the confusion matrix plotter.
        
        Args:
            model: Trained tensorflow model
            test_dataset: tf.data.Dataset containing test data
        """
        self.model = model
        self.test_dataset = test_dataset
        self.class_names = class_names
        
    def compute_confusion_matrix(self):
        """
        Compute the confusion matrix from model predictions.
        
        Returns:
            confusion matrix as numpy array
        """
        # Get predictions and true labels
        true_labels = []
        predictions = []
        
        # Iterate through the test dataset
        for images, labels in self.test_dataset:
            # Get predictions
            pred = self.model.predict(images, verbose=0)
            pred_classes = np.argmax(pred, axis=1)
            
            # Convert one-hot encoded labels back to class indices
            if len(labels.shape) > 1:  # if labels are one-hot encoded
                true_classes = np.argmax(labels, axis=1)
            else:
                true_classes = labels
                
            true_labels.extend(true_classes)
            predictions.extend(pred_classes)
            
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        return cm
    
    def plot_confusion_matrix(self, output_path, figsize=(12, 10)):
        """
        Create and save a confusion matrix plot.
        
        Args:
            output_path: Path where to save the confusion matrix plot
            figsize: Tuple of (width, height) for the figure size
        """
        # Compute confusion matrix
        cm = self.compute_confusion_matrix()
        
        # Create figure and axes
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        # Set labels and title
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()
        
    def plot_normalized_confusion_matrix(self, output_path, figsize=(12, 10)):
        """
        Create and save a normalized confusion matrix plot.
        
        Args:
            output_path: Path where to save the confusion matrix plot
            figsize: Tuple of (width, height) for the figure size
        """
        # Compute confusion matrix
        cm = self.compute_confusion_matrix()
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure and axes
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        # Set labels and title
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path)
        plt.close()

# Example usage:
"""
# After training your model:
confusion_plotter = ConfusionMatrixPlotter(model, test_dataset)

# Save regular confusion matrix
confusion_plotter.plot_confusion_matrix('confusion_matrix.png')

# Save normalized confusion matrix
confusion_plotter.plot_normalized_confusion_matrix('normalized_confusion_matrix.png')
""" 