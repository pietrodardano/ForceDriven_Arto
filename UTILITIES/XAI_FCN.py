import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import List, Union, Optional

def compute_grad_cam(model: Model, inputs: List[np.ndarray], last_layers_names: str, class_idx: int, epsilon: float = 1e-9) -> np.ndarray:
    model = Model(inputs=model.inputs, outputs=[model.get_layer(last_layers_names).output, model.output])
    with tf.GradientTape() as tape:
        output, predictions = model(inputs)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, output) #  list or nested structure of Tensors (or IndexedSlices, or None, or CompositeTensor), one for each element in sources.
    
    #reduced_grads == pooled_grads 
    reduced_grads = tf.reduce_mean(grads, axis=(0, 1))
    output = output[0] #otherwise it will remain a tensor, not a 1D 
    
    #make them workable by numpy 
    reduced_grads = reduced_grads.numpy()
    output = output.numpy()

    for i in range(reduced_grads.shape[-1]):
        output[:, i] *= reduced_grads[i]        #obtaining an heatmap
    
    heatmap = np.mean(output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0: heatmap /= np.max(heatmap) #to avoid zero division
    return heatmap

def plot_grad_cam(model: Model, 
                  X_tests: List[Union[np.ndarray, np.ndarray]], 
                  sample_idx: int, 
                  y_test: np.ndarray, 
                  conv_layers: List[str],
                  labels: Optional[List[str]] = None) -> None:
    """
    Plot Grad-CAM heatmaps for a given model and test samples.

    Args:
        model (Model): Trained Keras model.
        X_tests (List[np.ndarray]): List of test datasets corresponding to model inputs.
        sample_idx (int): Index of the sample to be visualized.
        y_test (np.ndarray): Ground truth labels for the test set.
        conv_layers (List[str]): List of convolutional layer names for which Grad-CAM will be computed.
        labels (Optional[List[str]]): List of labels for the signals. If None, default labels "Signal 1", "Signal 2", etc., will be used.

    Example usage:
        sample_idx = 6  # Change as needed
        X_tests = [X_test1, X_test2, X_test3]
        conv_layers = ['conv1d_1_3', 'conv1d_2_3', 'conv1d_3_3']
        plot_grad_cam(model, X_tests, sample_idx, y_test, conv_layers)
    """
    # Prepare inputs
    try:
        inputs = [np.expand_dims(X_test[sample_idx], axis=0) for X_test in X_tests]
    except IndexError as e:
        print(f"Error: {e}. Check if sample_idx {sample_idx} is within the range of X_tests.")
        return
    
    # Predict class index and probabilities
    y_pred_prob = model.predict(inputs)
    class_idx = np.argmax(y_pred_prob, axis=1)[0]
    y_pred_label = class_idx if y_pred_prob.shape[1] > 1 else (y_pred_prob[0][0] > 0.5).astype(int)

    # Compute Grad-CAM heatmaps
    heatmaps = [compute_grad_cam(model, inputs, conv_layer, class_idx) for conv_layer in conv_layers]

    # Plot Grad-CAM heatmaps
    plt.figure(figsize=(12, 3 * len(X_tests)))

    if labels is None:
        labels = [f'Branch {i+1}' for i in range(len(X_tests))]

    for i, (X_test, heatmap) in enumerate(zip(X_tests, heatmaps)):
        plt.subplot(len(X_tests), 1, i + 1)
        #plt.title(f'Grad-CAM for {labels[i]} signal')
        plt.title(f'Grad-CAM for {labels[i]}')
        
        if X_test.ndim == 2:  # Single channel input
            plt.plot(X_test[sample_idx])
            plt.imshow(heatmap[np.newaxis, :], aspect="auto", cmap='summer', alpha=0.6,
                       extent=(0, X_test.shape[1], np.min(X_test[sample_idx]), np.max(X_test[sample_idx])))
            plt.ylim(np.min(X_test[sample_idx]) - 0.0 * np.abs(np.min(X_test[sample_idx])), 
                     np.max(X_test[sample_idx]) + 0.0 * np.abs(np.max(X_test[sample_idx])))
            #plt.ylabel('[m]')
            if i == len(X_tests) - 1:  # Only show x-label for the last subplot
                plt.xlabel('Samples [2ms]')
        elif X_test.ndim == 3:  # Multi-channel input
            for channel in range(X_test.shape[-1]):
                plt.plot(X_test[sample_idx, :, channel], label=f'Signal {channel+1}')
            plt.imshow(heatmap[np.newaxis, :], aspect="auto", cmap='summer', alpha=0.6,
                       extent=(0, X_test.shape[1], np.min(X_test[sample_idx]), np.max(X_test[sample_idx])))
            plt.ylim(np.min(X_test[sample_idx]) - 0.0 * np.abs(np.min(X_test[sample_idx])), 
                     np.max(X_test[sample_idx]) + 0.0 * np.abs(np.max(X_test[sample_idx])))
            # if i == len(X_tests) - 1: 
            #     plt.ylabel('[m]')
            # else:
            #     plt.ylabel('[N]')
            if i == len(X_tests) - 1:  # Only show x-label for the last subplot
                plt.xlabel('Samples [2ms]')
            plt.legend()
        
        plt.colorbar()
    
    plt.suptitle(f"Test data number: {sample_idx} --> Yreal: {y_test[sample_idx]}, Ypred: {y_pred_label}")
    plt.tight_layout()
    plt.show()


# Example usage:
# sample_idx = 6  # Change as needed
# X_tests = [X_test1, X_test2, X_test3]
# conv_layers = ['conv1d_1_3', 'conv1d_2_3', 'conv1d_3_3']
# labels = ['Signal 1', 'Signal 2', 'Signal 3']
# plot_grad_cam(model, X_tests, sample_idx, y_test, conv_layers, labels)