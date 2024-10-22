import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
import time

def plot_data(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)
    
    first_pose_y = data.loc[0, 'Y']
    file_name = os.path.basename(file_path)
    
    # Plot settings
    fieldnames = ["Timestamp", "Pose_X", "Pose_Y", "Pose_Z", "Pose_Rx", "Pose_Ry", "Pose_Rz",
                  "Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z", "Y"]
    num_subplots = 2
    fig, axs = plt.subplots(num_subplots, figsize=(10, 6))
    
    # Plotting
    DposZ = np.abs(data['Pose_Z'][0]-data['Pose_Z'])
    axs[0].plot(data["Timestamp"], data["Force_Z"])
    axs[0].set_title('ForceZ')
    axs[1].plot(data["Timestamp"], DposZ)
    axs[1].set_title('DeltaPoseZ')
    plt.suptitle(f"File: {file_name}   Success -> {first_pose_y}")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def select_random_file(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return os.path.join(folder_path, random.choice(files))

def main(folder_path):
    files = [select_random_file(folder_path) for _ in range(10)]
    for file_path in files:
        plot_data(file_path)
        time.sleep(0.2)
        plt.close()

if __name__ == "__main__":
    folder_path = "/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/KNOB_OPEN"
    main(folder_path)
