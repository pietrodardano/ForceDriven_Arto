import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing CSV files
directory = '/home/user/arto_ws/src/RobotData_SIMA320/TESTGAIN'
#directory = '/home/rluser/thesis_ws/src/RobotData_ELITE/ARMPushTest'

# Function to read CSV files and plot 'Force_Z' data
def plot_force_z_data(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            plt.plot(df['Force_Z'], label=filename)
    
    plt.xlabel('Sample [2ms]')
    plt.ylabel('[N]')
    plt.title('Force_Z Data from CSV Files')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_force_z_data_subplots(directory):
    # Get list of CSV files
    csv_files = [filename for filename in os.listdir(directory) if filename.endswith(".csv")]
    
    # Calculate number of subplots required
    num_files = len(csv_files)
    num_cols = 2  # Number of columns for subplots
    num_rows = (num_files + num_cols - 1) // num_cols  # Calculate number of rows
    
    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6))
    
    # Flatten axs if only one row
    if num_rows == 1:
        axs = [axs]

    for i, filename in enumerate(csv_files):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        
        row = i // num_cols
        col = i % num_cols
        
        axs[row][col].plot(df['Force_Z'], label=filename)
        axs[row][col].set_xlabel('Index')
        axs[row][col].set_ylabel('Force_Z')
        axs[row][col].set_title(filename)
        axs[row][col].legend()
        axs[row][col].grid(True)

    # Remove empty subplots
    for i in range(len(csv_files), num_rows*num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row][col])

    plt.tight_layout()
    plt.show()

# Call the function
plot_force_z_data_subplots(directory)


# Call the function
#plot_force_z_data(directory)
