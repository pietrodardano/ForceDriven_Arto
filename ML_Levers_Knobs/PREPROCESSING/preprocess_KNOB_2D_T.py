import os
import pandas as pd
import numpy  as np

import sys
sys.path.append('/home/user/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

folder_path = "/home/user/thesis_ws/src/RobotData_SIMA320"
data_folder = '/home/user/thesis_ws/src/ML/DATA/2D_F_SIM/'

def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'Force_Z' in df.columns and 'Y' in df.columns:
            fz = df["Force_Z"] #.value
            
            #PREPROCESSING
            signal = myfilter(fz, cutoff_freq=30)
            res_half, res_std, res_rstd = num_transient(signal)
            first_good_index_half= sliding_sum_window(signal, squared_signal_start=res_half[0])
            first_good_index_std = sliding_sum_window(signal, squared_signal_start=res_std[0])
            first_good_index_rstd= sliding_sum_window(signal, squared_signal_start=res_rstd[0])
            start_signal_index = select_index(first_good_index_half, first_good_index_std, first_good_index_rstd, signal) #has distance samples in it
            end_signal_index = min(start_signal_index+WS_B, len(signal)) #redundancy
            sliced_signal = signal[start_signal_index : end_signal_index]

            # NORMALIZATION
            mean = np.mean(sliced_signal)
            normalized_signal= sliced_signal#/mean
            sliced_signal = add_padding(normalized_signal)
            
            coeffs = do_wavelet(sliced_signal, wavelet='morl')
            #print(coeffs.shape)
            
            y = df.loc[0, "Y"]
            X = coeffs
            return X, y
        else:
            print(f"Skipping file {csv_path}: Columns 'Force_Z' or 'Y' are missing.")
            rename_and_convert_to_txt(csv_path)
            return None, None
    except Exception as e:
        print(f"EXEPT: Error processing file {csv_path}: {e}")
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
#     X_data = np.transpose(X_data, (0, 2, 1))
    
#     return X_data, np.array(y_data)