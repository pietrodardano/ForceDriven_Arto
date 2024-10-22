import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time

import sys
sys.path.append('/home/rluser/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter

def plot_random_files(folder_path, num_files=15):
    """
    Plot the 'Force_Z' column from random CSV files in the specified folder and its subfolders.
    
    Parameters:
    - folder_path: path to the folder containing CSV files
    - num_files: number of random files to plot (default is 20)
    """
    # List to store file paths
    file_paths = []
    
    # Traverse the folder structure
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_paths.append(os.path.join(root, file))
    
    # Select random files
    random_files = random.sample(file_paths, num_files)
    
    # Plot from each random file
    for i, file_path in enumerate(random_files, 1):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if 'Force_Z' and 'Y' columns exist
            if 'Force_Z' in df.columns or 'Y' in df.columns:
                
                # Extract first element of 'Y' column
                #first_y_value = df['Y'].iloc[0]
                first_y_value = 1
                
                plt.figure(figsize=(10, 6))
                fz = df['Force_Z']
                
                # Plot 'Force_Z' column
                plt.plot(fz, label='Force_Z', color='blue')
                plt.plot(myfilter(fz, cutoff_freq= 30), label='Filtered 30')
                plt.plot(myfilter(fz, cutoff_freq= 15), label='Filtered 15')
                plt.plot(myfilter(fz, cutoff_freq= 8), label='Filtered 8')
                plt.axhline(np.mean(fz))
                # plt.xlim(0, 2000)
                # plt.ylim(-4, 4)

                # Get subfolder name
                subfolder_name = os.path.basename(os.path.dirname(file_path))
                
                plt.title(f'Subfolder: {subfolder_name} - Force_Z from {os.path.basename(file_path)} (Y = {first_y_value})')
                plt.xlabel('Sample')
                plt.ylabel('Force_Z')
                plt.legend()
                plt.grid(True)
                
                # Show plot
                plt.show()
                plt.close()
                
        except Exception as e:
            print(f"Error plotting {file_path}: {e}")

# Folder path
#folder_path = '/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/LEVER/'
# folder_path = '/home/rluser/thesis_ws/src/RobotData_GRIPA320/FCU_AP2'
# folder_path = '/home/rluser/thesis_ws/src/RobotData_SIMA320/AUTOBRAKE_between'
folder_path = '/home/rluser/thesis_ws/src/RobotData_ELITE/BrakePushTest/'

# Plot random files
plot_random_files(folder_path)
