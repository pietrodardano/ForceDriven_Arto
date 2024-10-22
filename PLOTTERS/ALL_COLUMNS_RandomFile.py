import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys
sys.path.append('/home/rluser/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter

# Define the fieldnames including "N"
fieldnames = ["Pose_X", "Pose_Y", "Pose_Z", "Pose_Rx", "Pose_Ry", "Pose_Rz",
              "Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z"] #,
              #"Q_w1", "Q_w2", "Q_w3"]


def calculate_y_limits(df, pattern):
    global_min = float('inf')
    global_max = float('-inf')
    for field in df.columns:
        if field.startswith(pattern):
            global_min = min(global_min, df[field].min())
            global_max = max(global_max, df[field].max())
    return global_min, global_max

def plot_csv(csv_file):
    df = pd.read_csv(csv_file)
    N = df['Y'].iloc[0] if 'Y' in df.columns else None
    num_cols = 2
    num_rows = (len(fieldnames) + 1) // num_cols
    
    # Calculate the differences for Pose_X, Pose_Y, Pose_Z
    differences = {}
    for field in ["Pose_X", "Pose_Y", "Pose_Z"]:
        if field in df.columns:
            F = df[field]
            differences[field] = F.max() - F.min()

    # Identify the field with the highest difference
    max_diff_field = max(differences, key=differences.get)

    # Calculate global min and max for each category
    
    force_min, force_max = calculate_y_limits(df, "Force_")
    torque_min, torque_max = calculate_y_limits(df, "Torque_")
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    target_length = 1800

    for i, field in enumerate(fieldnames):
        row = i // num_cols
        col = i % num_cols

        F = df[field]
        
        # Calculate the delta for Pose_* and Pose_R* fields
        if field.startswith("Pose_") or field.startswith("Pose_R"):
            F = F - F.iloc[0]
            

        filt_F = myfilter(F, cutoff_freq=30)
        
        if len(F) < target_length:
            padding_length = target_length - len(F)
            last_value = F.iloc[-1]
            padded_signal = np.pad(F, (0, padding_length), mode='constant', constant_values=last_value)
            if not field.startswith("Pose_R"):
                noise_mean = 0
                noise_std = np.std(F - filt_F)
                noise = np.random.normal(noise_mean, noise_std, padding_length)
                padded_signal[-padding_length:] += noise
        else:
            padded_signal = F
        
        axs[row, col].plot(padded_signal[:len(F)], label=field)
        if len(F) < target_length:
            axs[row, col].plot(range(len(F), target_length), padded_signal[len(F):], color='r')
        axs[row, col].set_title(field)
        axs[row, col].set_xlim(0, target_length)
        
        # Set y-limits based on field category and change background color
        if field.startswith("Pose_R"):
            #axs[row, col].set_ylim(-0.002, 0.002)
            axs[row, col].set_ylabel("[rad]")
            axs[row, col].set_facecolor('lightcyan')  # Change background color for Pose_R*
        elif field.startswith("Pose_"):
            axs[row, col].set_ylim(1.2*F.min(), 1.2*F.max())
            axs[row, col].set_ylabel("[m]")
        elif field.startswith("Force_"):
            axs[row, col].set_ylim(1.2*force_min, 1.2*force_max)
            axs[row, col].set_ylabel("[N]")
            axs[row, col].set_facecolor('aquamarine')  # Change background color for Force_*
        elif field.startswith("Torque_"):
            axs[row, col].set_ylim(1.2*torque_min, 1.2*torque_max)
            axs[row, col].set_ylabel("[Nm]")
            axs[row, col].set_facecolor('lavender')  # Change background color for Torque_*
            
        axs[row, col].legend()
        axs[row, col].grid(True)
        
        # Highlight the subplot with the highest difference
        if field == max_diff_field:
            axs[row, col].set_facecolor('lightyellow')

    folder_name = os.path.basename(os.path.dirname(csv_file))
    plt.suptitle(f"Subfolder: {folder_name} - CSV File: {os.path.basename(csv_file)} - N: {N}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Function to plot a single CSV file
def plot_csv_OLD(csv_file):
    df = pd.read_csv(csv_file)
    N = df['Y'].iloc[0] if 'Y' in df.columns else None
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    #plt.xlim(0, 3000)
    for i, field in enumerate(fieldnames):
        row = i // num_cols
        col = i % num_cols

        F=df[field]
        filt_F = myfilter(F, cutoff_freq=30)
        target_length = 2000
        if len(F) < target_length:
            padding_length = target_length - len(F)
            last_value = F.iloc[-1]
            padded_signal = np.pad(F, (0, padding_length), mode='constant', constant_values=last_value)
            noise_mean = 0
            noise_std =np.std(F-filt_F)
            noise = np.random.normal(noise_mean, noise_std, padding_length)
            padded_signal[-padding_length:] += noise
        else: 
            padded_signal = F
            
        axs[row, col].plot(padded_signal[:len(F)], label=field)
        if len(F)<target_length: axs[row, col].plot(range(len(F), target_length), padded_signal[len(F):], color='r')
        #axs[row, col].axhline(y=np.mean(df[field]), color='r')
        axs[row, col].set_ylabel(field)
        axs[row, col].set_xlim(0,2000)
        axs[row, col].legend()
        axs[row, col].grid(True)
    folder_name = os.path.basename(os.path.dirname(csv_file))
    plt.suptitle(f"Subfolder: {folder_name} - CSV File: {os.path.basename(csv_file)} - N: {int(N)}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Function to get 20 random CSV files
def get_random_files(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return random.sample(csv_files, min(20, len(csv_files)))

# Get 20 random CSV files
#random_files = get_random_files('/home/rluser/thesis_ws/src/RobotData_GRIPA320')
random_files = get_random_files('/home/rluser/thesis_ws/src/ROBOT_ACTIONS_DATA/FLAP')
#random_files = get_random_files('/home/rluser/thesis_ws/src/RobotData_GRIPA320')

# Calculate the number of rows and columns for subplots
num_cols = 2
num_rows = (len(fieldnames) + 1) // num_cols

# Plot each random CSV file
for csv_file in random_files:
    print(f"Plotting: {csv_file}")
    plot_csv(csv_file)
