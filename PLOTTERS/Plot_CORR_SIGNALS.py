import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec

def calculate_y_limits(df, prefix):
    columns = [col for col in df.columns if col.startswith(prefix)]
    if columns:
        min_val = df[columns].min().min()
        max_val = df[columns].max().max()
        return min_val, max_val
    return None, None

def plot_csv_and_correlation(csv_file):
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

    fig = plt.figure(figsize=(14, 18))  # Increased the figure size for better readability
    gs = gridspec.GridSpec(num_rows + 1, num_cols, height_ratios=[1] * num_rows + [2])  # Allocate more space for the correlation matrix

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

        ax = fig.add_subplot(gs[row, col])
        ax.plot(padded_signal[:len(F)], label=field)
        if len(F) < target_length:
            ax.plot(range(len(F), target_length), padded_signal[len(F):], color='r')
        ax.set_title(field)
        ax.set_xlim(0, target_length)

        # Set y-limits based on field category and change background color
        if field.startswith("Pose_"):
            ax.set_ylim(1.2 * F.min(), 1.2 * F.max())
            ax.set_ylabel("[m]")
        elif field.startswith("Pose_R"):
            ax.set_ylim(-0.002, 0.002)
            ax.set_ylabel("[rad]")
            ax.set_facecolor('lightcyan')  # Change background color for Pose_R*
        elif field.startswith("Force_"):
            ax.set_ylim(1.2 * force_min, 1.2 * force_max)
            ax.set_ylabel("[N]")
            ax.set_facecolor('aquamarine')  # Change background color for Force_*
        elif field.startswith("Torque_"):
            ax.set_ylim(1.2 * torque_min, 1.2 * torque_max)
            ax.set_ylabel("[Nm]")
            ax.set_facecolor('lavender')  # Change background color for Torque_*

        ax.legend()
        ax.grid(True)

        # Highlight the subplot with the highest difference
        if field == max_diff_field:
            ax.set_facecolor('lightyellow')

    # Compute the correlation matrix
    correlation_matrix = df[fieldnames].corr()

    # Plot the correlation matrix
    ax = fig.add_subplot(num_rows, num_cols, num_rows * num_cols)  # Create a subplot for correlation matrix
    cax = ax.matshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    #fig.colorbar(cax, ax=ax, orientation='horizontal')
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    ax.set_yticklabels(correlation_matrix.columns)
    ax.set_title(f"Correlation matrix for file: {os.path.basename(csv_file)}")

    # Annotate the heatmap with the correlation values
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def get_random_files(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return random.sample(csv_files, min(20, len(csv_files)))

# Example usage for checking 1D signals and plotting correlation matrices for 20 random CSV files:
folder_path = '/home/rluser/thesis_ws/src/ROBOT_ACTIONS_DATA/LDG'  # Adjust the path to your folder
random_files = get_random_files(folder_path)

# Plot each random CSV file for 1D signals and correlation matrix
for csv_file in random_files:
    print(f"Plotting 1D signals and correlation matrix for: {csv_file}")
    plot_csv_and_correlation(csv_file)
