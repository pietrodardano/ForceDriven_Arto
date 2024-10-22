import pandas as pd
import numpy as np
import os
import random
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def transform_force_torque_to_tcp(tcp_pose, force_base):
    """
    Transforms the force/torque from the base frame to the TCP frame.
    
    :param tcp_pose: A list containing the TCP position and orientation as a rotation vector [x, y, z, rx, ry, rz]
    :param force_base: A list containing the force/torque in the base frame [Fx, Fy, Fz, Tx, Ty, Tz]
    :return: Transformed force/torque in the TCP frame
    """
    # Extract the rotation vector from the TCP pose
    rotation_vector = np.array(tcp_pose[3:])
    
    # Convert rotation vector to rotation matrix
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    
    # Split the force and torque components
    force_base_vector = np.array(force_base[:3])
    torque_base_vector = np.array(force_base[3:])
    
    # Transform the force and torque to the TCP frame
    force_tcp_vector = np.dot(rotation_matrix.T, force_base_vector)
    torque_tcp_vector = np.dot(rotation_matrix.T, torque_base_vector)
    
    return np.concatenate((force_tcp_vector, torque_tcp_vector))

def get_random_csv_file(folder_path):
    """
    Randomly selects a CSV file from the provided folder path or from a random subfolder if subfolders exist.
    
    :param folder_path: Path to the folder containing CSV files or subfolders with CSV files.
    :return: Full path to a randomly selected CSV file, and the subfolder name (if any).
    """
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    
    if subfolders:
        selected_subfolder = random.choice(subfolders)
        csv_files = [f for f in os.listdir(selected_subfolder) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in subfolder: {selected_subfolder}")
        selected_csv = random.choice(csv_files)
        return os.path.join(selected_subfolder, selected_csv), os.path.basename(selected_subfolder), selected_csv
    else:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in folder: {folder_path}")
        selected_csv = random.choice(csv_files)
        return os.path.join(folder_path, selected_csv), '', selected_csv

def process_and_plot_csv(file_path, subfolder_name, file_name):
    """
    Processes the given CSV file, transforms the force/torque data, and plots the results.
    
    :param file_path: Path to the CSV file.
    :param subfolder_name: Name of the subfolder containing the file.
    :param file_name: Name of the CSV file.
    """
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Prepare the transformed data container
    transformed_data = []

    # Loop through each row to perform the transformation
    for index, row in data.iterrows():
        tcp_pose = [row['Pose_X'], row['Pose_Y'], row['Pose_Z'], row['Pose_Rx'], row['Pose_Ry'], row['Pose_Rz']]
        force_base = [row['Force_X'], row['Force_Y'], row['Force_Z'], row['Torque_X'], row['Torque_Y'], row['Torque_Z']]
        
        # Transform the forces and torques
        transformed_force_torque = transform_force_torque_to_tcp(tcp_pose, force_base)
        
        transformed_data.append(transformed_force_torque)

    # Convert transformed data to DataFrame for easier plotting
    transformed_df = pd.DataFrame(transformed_data, columns=['Force_X_TCP', 'Force_Y_TCP', 'Force_Z_TCP', 'Torque_X_TCP', 'Torque_Y_TCP', 'Torque_Z_TCP'])

    # Plotting the original vs transformed forces
    plt.figure(figsize=(12, 8))

    # Plot Forces in X, Y, Z
    plt.subplot(3, 2, 1)
    plt.plot(data['Force_X'], label='Force_X_Base')
    plt.plot(transformed_df['Force_X_TCP'], label='Force_X_TCP')
    plt.legend()
    plt.title('Force X')

    plt.subplot(3, 2, 3)
    plt.plot(data['Force_Y'], label='Force_Y_Base')
    plt.plot(transformed_df['Force_Y_TCP'], label='Force_Y_TCP')
    plt.legend()
    plt.title('Force Y')

    plt.subplot(3, 2, 5)
    plt.plot(data['Force_Z'], label='Force_Z_Base')
    plt.plot(transformed_df['Force_Z_TCP'], label='Force_Z_TCP')
    plt.legend()
    plt.title('Force Z')

    # Plot Torques in X, Y, Z
    plt.subplot(3, 2, 2)
    plt.plot(data['Torque_X'], label='Torque_X_Base')
    plt.plot(transformed_df['Torque_X_TCP'], label='Torque_X_TCP')
    plt.legend()
    plt.title('Torque X')

    plt.subplot(3, 2, 4)
    plt.plot(data['Torque_Y'], label='Torque_Y_Base')
    plt.plot(transformed_df['Torque_Y_TCP'], label='Torque_Y_TCP')
    plt.legend()
    plt.title('Torque Y')

    plt.subplot(3, 2, 6)
    plt.plot(data['Torque_Z'], label='Torque_Z_Base')
    plt.plot(transformed_df['Torque_Z_TCP'], label='Torque_Z_TCP')
    plt.legend()
    plt.title('Torque Z')

    plt.suptitle(f"Subfolder: {subfolder_name} | File: {file_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Main logic to process up to 15 random CSV files
folder_path = '/home/rluser/thesis_ws/src/RobotData_GRIPA320/OH_TEST/'  # Replace with your actual folder path

for i in range(15):
    try:
        file_path, subfolder_name, file_name = get_random_csv_file(folder_path)
        print(f"Processing file {i+1}: {file_path}")
        process_and_plot_csv(file_path, subfolder_name, file_name)
    except ValueError as e:
        print(e)
        break
