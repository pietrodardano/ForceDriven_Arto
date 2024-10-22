import os
import numpy as np
import matplotlib.pyplot as plt
import random

def plot_X_from_npz(file_path):
    try:
        # Load data from the .npz file
        data = np.load(file_path)
        X = data['X']
        y = data['y']
        
        # Plot the first row of X
        plt.plot(X)
        
        # Set labels and title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'X.npy Plot\nY: {y.item()}')
        plt.show()
        
        # Close the npz file
        data.close()

    except Exception as e:
        print(f"An error occurred: {e}")

#folder_path = '/home/user/thesis_ws/src/ML_Levers_Knobs/DATA/1D_LEVER_Fx'
data_folder = '/home/user/thesis_ws/src/ML_Levers_Knobs/DATA/test_padding'
npz_files = [file for file in os.listdir(data_folder) if file.endswith('.npz')]
num_files_to_plot = min(len(npz_files), 20)

random_npz_files = random.sample(npz_files, num_files_to_plot)

for file_name in random_npz_files:
    file_path = os.path.join(data_folder, file_name)
    plot_X_from_npz(file_path)




