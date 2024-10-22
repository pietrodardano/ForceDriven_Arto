import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Function to plot the "Force_Z" column from a CSV file
def plot_force_z(csv_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Check if "Force_Z" column exists in the DataFrame
    if 'Force_Z' not in df.columns:
        print("Error: 'Force_Z' column not found in the CSV file.")
        return

    ys = df.loc[0, 'Y']
    force_z = df['Force_Z']
    pose_x = np.abs(df['Pose_X'][0]-df['Pose_X'])
    pose_y = np.abs(df['Pose_Y'][0]-df['Pose_Y'])
    pose_z = np.abs(df['Pose_Z'][0]-df['Pose_Z'])

    # Filtering Force_Z signal
    nyq_freq = 0.5 * 500

    normalized_cutoff = 30 / nyq_freq
    b, a = butter(2, normalized_cutoff, btype='low')
    filtered_force_z = filtfilt(b, a, force_z)
    filtered_force_z[filtered_force_z < 0] = 0

    normalized_cutoff = 15 / nyq_freq
    b, a = butter(2, normalized_cutoff, btype='low')
    pose_xf = filtfilt(b, a, pose_x)
    pose_yf = filtfilt(b, a, pose_y)
    pose_zf = filtfilt(b, a, pose_z)

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(force_z, label='Force_Z')
    axs[0].plot(filtered_force_z, label='Filtered Force_Z')
    axs[0].axhline(y=np.std(filtered_force_z), color='r', linestyle='--', label='STD')
    axs[0].axhline(y=np.std(filtered_force_z) * 0.82, color='g', linestyle='--', label='RSTD')
    axs[0].axhline(y=np.max(filtered_force_z) / 2, color='b', linestyle='--', label='HALF')
    axs[0].legend()
    axs[0].set_ylabel('Force_Z')
    axs[0].set_xlim(0, 2000)

    Y_LIM = 0.002

    axs[1].plot(pose_x, color='orange', label='Pose_X')
    axs[1].plot(pose_xf, color='magenta')
    axs[1].legend()
    axs[1].set_ylabel('Pose_X')
    axs[1].set_ylim(0, Y_LIM)
    axs[1].set_xlim(0, 2000)

    axs[2].plot(pose_y)
    axs[2].plot(pose_yf, color='magenta', label='Pose_Y')
    axs[2].legend()
    axs[2].set_ylabel('Pose_Y')
    axs[2].set_ylim(0, Y_LIM)
    axs[2].set_xlim(0, 2000)

    axs[3].plot(pose_z, color='cyan', label='Pose_Z')
    axs[3].plot(pose_zf, color='magenta')
    axs[3].legend()
    axs[3].set_ylabel('Pose_Z')
    axs[3].set_ylim(0, Y_LIM)
    axs[3].set_xlim(0, 2000)

    plt.suptitle(f'File: {csv_path}, S: {ys}')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()

    # # Plot the original data in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(pose_xf, pose_yf, pose_zf)
    # ax.set_xlabel('PoseX')
    # ax.set_ylabel('PoseY')
    # ax.set_zlabel('PoseZ')
    # ax.set_title('Original Data')
    # # Set axis limits
    # ax.set_xlim(0, 0.012)
    # ax.set_ylim(0, 0.012)
    # ax.set_zlim(0, 0.012)
    # plt.show()

    # # Perform PCA
    # pose_data = np.array([pose_xf, pose_yf, pose_zf]).T
    # pca = PCA(n_components=1)
    # pca_data = pca.fit_transform(pose_data)

    # # Plot the data after PCA
    # plt.plot(pca_data)
    # plt.xlabel('Index')
    # plt.ylabel('Principal Component 1')
    # plt.title('Data After PCA')
    # plt.show()
    

def plot_random_files(csv_path, num_files=15):
    # Get a list of all CSV files in the subfolders
    csv_files = []
    for root, _, files in os.walk(csv_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    # Select 15 random files
    random_files = random.sample(csv_files, min(num_files, len(csv_files)))

    # Plot each random file
    for file_path in random_files:
        plot_force_z(file_path)

# Usage example:
csv_path = "/home/rluser/thesis_ws/src/ROBOT_ACTIONS_DATA/KNOB_OPEN"
plot_random_files(csv_path)