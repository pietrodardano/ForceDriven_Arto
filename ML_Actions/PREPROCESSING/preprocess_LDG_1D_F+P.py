import os
import pandas as pd
import numpy  as np

import sys
sys.path.append('/home/user/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet, pad_signal_with_noise
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA

data_folder = '/home/user/thesis_ws/src/ML_Levers_Knobs/DATA/1D_LEVER_Fx/'
folder_path = "/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/LEVER/"

nyq_freq = 0.5 * 500
normalized_cutoff = 30 / nyq_freq
b, a = butter(2, normalized_cutoff, btype='low')

def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'Force_X' in df.columns and 'Y' in df.columns and df['Pose_X'].nunique() != 1:
            fz = df["Force_X"]
            flag = 0

            pXs, pYs, pZs = np.array(df['Pose_X']), \
                            np.array(df['Pose_Y']), \
                            np.array(df['Pose_Z'])
            
            # PREPROCESSING
            target_length = 2000
            if len(fz) < target_length:
                padding_length = target_length - len(fz)
                last_value = fz.iloc[-1]
                last_value_X = pXs[-1]
                last_value_Y = pYs[-1]
                last_value_Z = pZs[-1]
                padded_signal   = np.pad(fz,  (0, padding_length), mode='constant', constant_values=last_value)
                padded_Xs       = np.pad(pXs, (0, padding_length), mode='constant', constant_values=last_value_X)
                padded_Ys       = np.pad(pYs, (0, padding_length), mode='constant', constant_values=last_value_Y)
                padded_Zs       = np.pad(pZs, (0, padding_length), mode='constant', constant_values=last_value_Z)
                flag = 1
            else:
                padded_signal = fz
                padded_Xs = pXs
                padded_Ys = pYs
                padded_Zs = pZs


            # Apply filtering
            filt_signal = myfilter(padded_signal, cutoff_freq=30)
            print(f"Original: {len(fz)}, Padded: {len(filt_signal)}, FLAG={flag}")

            normalized_cutoff = 15 / nyq_freq
            b, a = butter(2, normalized_cutoff, btype='low')
            # Calculate position differences using the padded positions
            dpos_X, dpos_Y, dpos_Z = np.abs(padded_Xs[0] - padded_Xs), np.abs(padded_Ys[0] - padded_Ys), np.abs(padded_Zs[0] - padded_Zs)
            dpos_X, dpos_Y, dpos_Z = filtfilt(b, a, dpos_X), filtfilt(b, a, dpos_Y), filtfilt(b, a, dpos_Z)

            # NORMALIZATION
            mean = np.mean(filt_signal)
            normalized_signal = filt_signal  # / mean

            X = np.dstack((normalized_signal, dpos_X, dpos_Y, dpos_Z))

            y = df.loc[0, "Y"]
            return X, y
        else:
            print(f"Skipping file {csv_path}: Columns 'Force_X' or 'Y' are missing.")
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