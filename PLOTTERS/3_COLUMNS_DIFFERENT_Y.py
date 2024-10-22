import os
import pandas as pd
import matplotlib.pyplot as plt
import random

def plot_force_torque(subfolder_path):
    # Define the columns to plot
    columns_to_plot = ['Force_X', 'Force_Z', 'Torque_Z']
    
    # Initialize a dictionary to hold dataframes for each Y value
    data_by_y = {0: [], 1: [], 2: [], 3: []}
    
    # Traverse the subfolder and read CSV files
    for file_name in os.listdir(subfolder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(subfolder_path, file_name)
            df = pd.read_csv(file_path)
            
            # Check if required columns are in the dataframe
            if 'Y' in df.columns and all(col in df.columns for col in columns_to_plot):
                y_value = df['Y'].iloc[0]
                if y_value in data_by_y:
                    data_by_y[y_value].append((df, file_name))
    
    # Filter out Y values without any data
    data_by_y = {y: data_list for y, data_list in data_by_y.items() if data_list}
    
    # If no data found, exit the function
    if not data_by_y:
        print(f"No valid data found in {subfolder_path}")
        return
    
    # Create a figure with subplots for each Y value
    num_rows = len(data_by_y)
    num_cols = len(columns_to_plot)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True)
    fig.suptitle(f'Data from {os.path.basename(subfolder_path)}', fontsize=16)
    
    # Plot a random file for each Y value
    for row, (y_value, data_list) in enumerate(data_by_y.items()):
        df, file_name = random.choice(data_list)
        for col, column_name in enumerate(columns_to_plot):
            ax = axs[row, col] if num_rows > 1 else axs[col]
            ax.plot(df[column_name], label=f'{column_name}')
            ax.set_ylabel(column_name)
            ax.grid(True)
            ax.legend()
            if col == 1:  # Middle column
                ax.set_title(f'Y={y_value}, File={file_name}')
    
    #axs[0, num_cols//2].set_xlabel('Sample Index')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    folder_path = '/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/LDG'
    
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            plot_force_torque(subfolder_path)

