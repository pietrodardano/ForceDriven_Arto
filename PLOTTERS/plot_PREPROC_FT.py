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

def plot_channels(X_data):
    num_channels_to_plot = 6
    num_subplots = num_channels_to_plot // 2 + num_channels_to_plot % 2  # 2 columns of subplots

    fig, axes = plt.subplots(num_subplots, 2, figsize=(12, 2*num_subplots))

    for i in range(num_channels_to_plot):
        row = i // 2
        col = i % 2
        axes[row, col].plot(X_data[0,:, i])
        axes[row, col].set_title(f'Channel {i+1}')
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel('Amplitude')

    # Adjust layout
    fig.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    data_folder = '/home/rluser/thesis_ws/src/ML_Levers_Knobs/DATA/1D_KNOB_FTP_ScalNorm'
    X_data, y_data = load_data(data_folder)

    # Assuming X_data.shape = (num_files, 2000, 9), where 9 is the number of channels
    # Plot the first 6 channels
    for i in range(0,500, 50):
        plot_channels(X_data[i])  # Plot channels from the first file, adjust as needed
