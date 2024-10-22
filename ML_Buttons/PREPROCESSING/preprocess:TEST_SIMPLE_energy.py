import os
import pandas as pd
import numpy  as np
import time

import sys
sys.path.append('/home/rluser/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from sklearn.preprocessing import StandardScaler

data_folder = '/home/rluser/thesis_ws/src/ML/DATA/1D_fullgripa_F/'

global flag
flag = 0

folder_path = "/home/rluser/thesis_ws/src/RobotData_GRIPA320"
def preprocess_data(csv_path):
    global flag  # Add this line to access the global flag variable
    try:
        df = pd.read_csv(csv_path)
        if 'Force_Z' in df.columns and 'Y' in df.columns:
            fz = df["Force_Z"]
            if flag <=5:
                start = time.time()

            # PREPROCESSING
            signal = myfilter(fz, cutoff_freq=30)

            # Energy Detection Logic
            window_size = 300
            energy = np.array([np.sum(signal[i:i+window_size]**2) for i in range(len(signal) - window_size + 1)])
            max_energy = np.max(energy)
            thresholds = [0.55, 0.45, 0.35]  # Thresholds as percentages of max energy
            
            indices = []
            for threshold in thresholds:
                for i, e in enumerate(energy):
                    if e >= max_energy * threshold:
                        indices.append(i)
                        break

            if len(indices) < 3:
                print(f"Error: Less than 3 indices found for file {csv_path}")
                return None, None

            # Check if the indices are close enough (< 60 samples from the middle one)
            if abs(indices[2] - indices[1]) < 60 and abs(indices[0] - indices[1]) < 60:
                start_signal_index = indices[1]
            else:
                start_signal_index = indices[0]

            end_signal_index = min(start_signal_index + window_size, len(signal))
            sliced_signal = add_padding(signal[start_signal_index:end_signal_index])
            
            mean = np.mean(sliced_signal)

            # NORMALIZATION
            # if mean > 1 or mean <-1:
            #     normalized_signal = sliced_signal  / mean
            # else:
            #     normalized_signal = sliced_signal

            # Not Normalized
            # normalized_signal = sliced_signal

            if flag <=5 :
                end = time.time()
                print(f"[TIME] -- Preprocessing time: {end-start} \n")
                time.sleep(5)
                flag += 1

            # SCALED NORMALIZATION
            scaler_Fx = StandardScaler()
            normalized_signal = scaler_Fx.fit_transform(sliced_signal.reshape(-1, 1)).flatten()

            if np.mean(normalized_signal)<0 : normalized_signal = -normalized_signal
            y = df.loc[0, "Y"]
           

            return normalized_signal, y
        else:
            print(f"Skipping file {csv_path}: Columns 'Force_Z' or 'Y' are missing.")
            rename_and_convert_to_txt(csv_path)
            return None, None
    except Exception as e:
        print(f"Error processing file {csv_path}: {e}")
        return None, None

def preprocess_folder_data(folder_path, data_folder):
    # Create the data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    c=0
    
    # Traverse the folder structure
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                X_data, y_data = preprocess_data(file_path)
                if X_data is not None and y_data is not None:
                    # Save preprocessed data into a file in the data folder
                    file_name = os.path.splitext(file)[0] + f"#{c}_preprocessed.npy"
                    save_path = os.path.join(data_folder, file_name)
                    np.savez(save_path, X=X_data, y=y_data)
                    print(f"Preprocessed data saved to {save_path}")
                    c +=1
                    #print("num --> ", c)

# Call the preprocess_folder_data function
preprocess_folder_data(folder_path, data_folder)

# # Function to navigate through directory and preprocess data from CSV files
# def preprocess_folder_data(folder_path):
#     X_data = []
#     y_data = []
#     # Traverse the folder structure
#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(".csv"):
#                 file_path = os.path.join(root, file)
#                 X, y = preprocess_data(file_path)
#                 if X is not None and y is not None and len(X) >= 10 and X.ndim != 0:
#                     print(f"{file_path[len(folder_path)+1:]} adding, its len: {X.shape}, Y is: {y}")
#                     X_data.append(X)
#                     y_data.append(y)
#                     print("current X_data: ", len(X_data), len(X_data[len(X_data)-1]))
#     X_data = np.array(X_data)
    
#     return X_data, np.array(y_data)