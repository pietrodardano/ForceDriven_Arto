import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def calculate_y_limits(df, prefix):
    columns = [col for col in df.columns if col.startswith(prefix)]
    if columns:
        min_val = df[columns].min().min()
        max_val = df[columns].max().max()
        return min_val, max_val
    return None, None

def plot_csv(csv_file):
    df = pd.read_csv(csv_file)
    fieldnames = ["Pose_X", "Pose_Y", "Pose_Z", "Pose_Rx", "Pose_Ry", "Pose_Rz",
                  "Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z"]
    
    num_cols = 2
    num_rows = (len(fieldnames) + 1) // num_cols
    
    # Calculate the differences for Pose_X, Pose_Y, Pose_Z
    differences = {field: df[field].max() - df[field].min() for field in ["Pose_X", "Pose_Y", "Pose_Z"] if field in df.columns}

    # Identify the field with the highest difference
    max_diff_field = max(differences, key=differences.get)

    # Calculate global min and max for each category
    force_min, force_max = calculate_y_limits(df, "Force_")
    torque_min, torque_max = calculate_y_limits(df, "Torque_")
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    target_length = 3000

    for i, field in enumerate(fieldnames):
        row = i // num_cols
        col = i % num_cols

        F = df[field]
        
        # Calculate the delta for Pose_* and Pose_R* fields
        if field.startswith("Pose_") or field.startswith("Pose_R"):
            F = F - F.iloc[0]
        
        if len(F) < target_length:
            padding_length = target_length - len(F)
            last_value = F.iloc[-1]
            padded_signal = np.pad(F, (0, padding_length), mode='constant', constant_values=last_value)
            noise_mean = 0
            noise_std = np.std(F - F.rolling(window=30, min_periods=1).mean())
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
        if field.startswith("Pose_"):
            axs[row, col].set_ylim(1.2*F.min(), 1.2*F.max())
            axs[row, col].set_ylabel("[m]")
        elif field.startswith("Pose_R"):
            axs[row, col].set_ylim(-0.002, 0.002)
            axs[row, col].set_ylabel("[rad]")
            axs[row, col].set_facecolor('lightcyan')  # Change background color for Pose_R*
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
    plt.suptitle(f"Subfolder: {folder_name} - CSV File: {os.path.basename(csv_file)}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix_csv(csv_file):
    df = pd.read_csv(csv_file)
    fieldnames = ["Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z", "Pose_X", "Pose_Y", "Pose_Z"]
    dfy = df
    df = df[fieldnames]
    
    # Compute the correlation matrix
    correlation_matrix = df.corr()
    
    # Plot the correlation matrix using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title(f"Correlation matrix for file: {os.path.basename(csv_file)}, Y = {dfy['Y'][0]}")

    # Annotate the heatmap with the correlation values
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()

def get_random_files(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return random.sample(csv_files, min(5, len(csv_files)))

# Example usage for checking 1D signals and plotting correlation matrices for 20 random CSV files:
folder_path = '/home/rluser/thesis_ws/src/ROBOT_ACTIONS_DATA/FLAP'  # Adjust the path to your folder
#folder_path = "/home/rluser/thesis_ws/src/ROBOT_ACTIONS_DATA/KNOB_OPEN"
#folder_path = "/home/rluser/thesis_ws/src/RobotData_GRIPA320"
random_files = get_random_files(folder_path)

# Plot each random CSV file for 1D signals
# for csv_file in random_files:
#     print(f"Plotting 1D signals for: {csv_file}")
#     #plot_csv(csv_file)

# Plot each random CSV file for correlation matrix
for csv_file in random_files:
    print(f"Plotting correlation matrix for: {csv_file}")
    plot_correlation_matrix_csv(csv_file)


# /home/rluser/thesis_ws/src/RobotData_GRIPA320/OH_HOTAIR/Butt_7N_1000_#52.csv