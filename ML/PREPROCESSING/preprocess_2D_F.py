import os
import pandas as pd
import numpy  as np

from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('/home/rl_sim/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

folder_path = "/home/rl_sim/thesis_ws/src/RobotData_GRIPA320"
data_folder = '/home/rl_sim/thesis_ws/src/ML/DATA/2D_F_GRIP_ScalNorm/'

WS_B = 800

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
            # if mean > 1 or mean <-1:
            #     normalized_signal = sliced_signal  / mean
            # else:
            #     normalized_signal = sliced_signal

            # Not Normalized
            # normalized_signal = sliced_signal

            # SCALED NORMALIZATION
            scaler_Fx = StandardScaler()
            normalized_signal = scaler_Fx.fit_transform(sliced_signal.reshape(-1, 1)).flatten()

            if np.mean(normalized_signal)<0 : normalized_signal = -normalized_signal

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
