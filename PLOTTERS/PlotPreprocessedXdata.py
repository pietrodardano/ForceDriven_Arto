import os
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_folder):
    X_data_list = []
    y_data_list = []
    # Get list of all .npz files in the data folder
    npz_files = [file for file in os.listdir(data_folder) if file.endswith('.npz')]
    
    # Load data from each .npz file
    for file in npz_files:
        file_path = os.path.join(data_folder, file)
        with np.load(file_path) as data:
            X_data_list.append(data['X'])  # Assuming X_data is stored under 'X' key in the .npz file
            y_data_list.append(data['y'])  # Assuming y_data is stored under 'y' key in the .npz file
    
    # Convert lists to NumPy arrays
    X_data = np.array(X_data_list)
    y_data = np.array(y_data_list)
    
    return X_data, y_data


def plot_negative_mean_signals(X_data):
    for i, X in enumerate(X_data):
        # Plot the data if the mean is negative
        plt.figure(figsize=(12, 6))
        
        # First subplot: original signal
        plt.subplot(1, 2, 1)
        plt.plot(X[:,1])
        plt.ylim(-10, 10)
        plt.axhline(y=0, color='r', linestyle='--')  # Add red horizontal line at y=0
        plt.title(f'Original Sample {i}')
        plt.grid(True)
        
        # Second subplot: mirrored signal
        plt.subplot(1, 2, 2)
        plt.plot(-X[:,1])
        plt.ylim(-10, 10)
        plt.axhline(y=0, color='r', linestyle='--')  # Add red horizontal line at y=0
        plt.title(f'Mirrored Sample {i}')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        # Wait for the plot to be closed before continuing to the next one
        plt.close()


folder_directory = '/home/rluser/thesis_ws/src/ML_Levers_Knobs/DATA/1D_AUGM_KNOB_FTP_ScalNorm'  
X_data, y_data = load_data(folder_directory)
plot_negative_mean_signals(X_data)
