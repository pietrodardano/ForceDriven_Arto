import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt

def plot_random_files(folder_path, folders, duration=20):
    # Plot files from each selected folder for the specified duration
    start_time = time.time()
    while time.time() - start_time < duration:
        for folder in folders:
            # Get a list of files in the folder
            folder_files = os.listdir(os.path.join(folder_path, folder))
            for i in range(3):
                # Select a random file
                file = random.choice(folder_files)
                
                # Read the CSV file and extract the columns
                file_path = os.path.join(folder_path, folder, file)
                df = pd.read_csv(file_path)
                force_z = df['Force_Z']
                pose_x = np.abs(df['Pose_X'][0]-df['Pose_X'])
                pose_y = np.abs(df['Pose_Y'][0]-df['Pose_Y'])
                pose_z = np.abs(df['Pose_Z'][0]-df['Pose_Z'])
                
                # Filtering Force_Z signal
                nyq_freq = 0.5 * 500
                normalized_cutoff = 25 / nyq_freq
                b, a = butter(2, normalized_cutoff, btype='low')
                filtered_force_z = filtfilt(b, a, force_z)
                filtered_force_z[filtered_force_z < 0] = 0
                pose_yf = filtfilt(b, a, pose_y)
                
                # Plotting
                fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                axs[0].plot(force_z, label='Force_Z')
                axs[0].plot(filtered_force_z, label='Filtered Force_Z')
                axs[0].axhline(y=np.std(filtered_force_z), color='r', linestyle='--', label='STD')
                axs[0].axhline(y=np.std(filtered_force_z) * 0.82, color='g', linestyle='--', label='RSTD')
                axs[0].axhline(y=np.max(filtered_force_z) / 2, color='b', linestyle='--', label='HALF')
                axs[0].legend()
                axs[0].set_ylabel('Force_Z')
                axs[0].set_title(f'{folder} -> {file}')
                
                axs[1].plot(pose_x, color='orange', label='Pose_X')
                axs[1].legend()
                axs[1].set_ylabel('Pose_X')
                
                axs[2].plot(pose_y)
                axs[2].plot(pose_yf, color='magenta', label='Pose_Y')
                axs[2].legend()
                axs[2].set_ylabel('Pose_Y')
                
                axs[3].plot(pose_z, color='cyan', label='Pose_Z')
                axs[3].legend()
                axs[3].set_ylabel('Pose_Z')
                
                plt.xlabel('Time')
                plt.tight_layout()
                plt.show()
                
                #Pause for a short interval between plots
                #time.sleep(1)
                plt.close()               

# Example usage:
# Assuming you have a directory 'data' containing multiple folders each with CSV files
folder_path = '/home/user/arto_ws/src/RobotData_SIMA320/'
#folders = ['ALTPushTest', 'BlackButtPushTest', 'BrakePushTest', 'TestSim', 'YellowArrowPushTest', 'ARMPushTest', 'B_Push', 'FramePushTest', 'WhiteRelePushTest']  # List of folders to plot from
folders = ["LS_over_but", "MCDU_init", "TERR_but", "TERR_brd_but", "MCDU_5",
              "MCDU_frame", "MCDU_CLR", "MCDU_L", "ECAM_FCTL", "ECAM_wheel",
              "LS_frame", "TERR_frame", "MCDU_prog", "AUTOBRAKE", "AUTOBRAKE_between", "VH_greenArrow"]
#folder_path = '/home/user/arto_ws/src/RobotData_GRIPA320'
#folders = ['TERR_but']
plot_random_files(folder_path, folders)

# /home/pietro/arto_ws/src/RobotData_SIMA320