import os
import pandas as pd
import matplotlib.pyplot as plt
import random

def plot_random_files(folder_path, num_files=20):
    """
    Plot the 'Force_Z' column from random CSV files in the specified folder and its subfolders, 
    along with the Delta Pose_X, Delta Pose_Y, and Delta Pose_Z columns.
    
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
            
            # Check if 'Force_X', 'Force_Y', 'Pose_X', 'Pose_Y', and 'Pose_Z' columns exist
            if all(col in df.columns for col in ['Force_X', 'Force_Y', 'Pose_X', 'Pose_Y', 'Pose_Z']):
                

                first_y_value = df['Y'].iloc[0]
                # Calculate delta Pose_X, Pose_Y, Pose_Z
                delta_pose_x = df['Pose_X'].iloc[0] - df['Pose_X']
                delta_pose_y = df['Pose_Y'].iloc[0] - df['Pose_Y']
                delta_pose_z = df['Pose_Z'].iloc[0] - df['Pose_Z']
                
                # Get subfolder name
                subfolder_name = os.path.basename(os.path.dirname(file_path))
                
                # Plotting
                plt.figure(figsize=(15, 10))
                
                # Plot Force_Z
                plt.subplot(3, 2, 1)
                plt.plot(df['Force_Z'], color='blue')
                plt.title('Force_Z')
                plt.grid(True)
                
                # Plot Force_X
                plt.subplot(3, 2, 2)
                plt.plot(df['Force_X'], color='blue')
                plt.title('Force_X')
                plt.grid(True)
                
                # Plot Force_Y
                plt.subplot(3, 2, 3)
                plt.plot(df['Force_Y'], color='red')
                plt.title('Force_Y')
                plt.grid(True)

                # Plot Delta Pose_X
                plt.subplot(3, 2, 4)
                plt.plot(delta_pose_x, color='green', linestyle='--')
                plt.title('Delta Pose_X')
                plt.grid(True)

                # Plot Delta Pose_Y
                plt.subplot(3, 2, 5)
                plt.plot(delta_pose_y, color='orange', linestyle='--')
                plt.title('Delta Pose_Y')
                plt.grid(True)

                # Plot Delta Pose_Z
                plt.subplot(3, 2, 6)
                plt.plot(delta_pose_z, color='purple', linestyle='--')
                plt.title('Delta Pose_Z')
                plt.grid(True)
                
                plt.suptitle(f'Subfolder: {subfolder_name} - Data from {os.path.basename(file_path)}, (Y = {first_y_value})')
                plt.tight_layout()
                
                # Show plot
                plt.show()
                plt.close()
                
        except Exception as e:
            print(f"Error plotting {file_path}: {e}")

# Folder path
folder_path = '/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/FLAP'

# Plot random files
plot_random_files(folder_path)
