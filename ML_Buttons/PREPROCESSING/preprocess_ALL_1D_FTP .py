import os
import pandas as pd
import numpy  as np

import sys
sys.path.append('/home/rl_sim/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet, pad_signal_with_noise
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

data_folder = '/home/rl_sim/thesis_ws/src/ML/DATA/1D_GRIPA_FTP_NotNorm/'
folder_path = "/home/rl_sim/thesis_ws/src/RobotData_GRIPA320"

def preprocess_signal(signal, cutoff_freq=30, start=0, end=2000, tonorm=0):
    signal = myfilter(signal, cutoff_freq)
    sliced_signal = signal[start: end]
    if end-start != WS_B: print("!! -- ERROR -- !! end-start not ws_b")

    filt_signal = myfilter(sliced_signal, cutoff_freq)

    mean = np.mean(filt_signal)
    if tonorm == 1 and mean != 0:     
        #  MEAN NORMALIZATION
        normalized_signal = filt_signal /mean
    elif tonorm == 2:
        # NORMALIZATION using StandardScaler
        signal_scaler = StandardScaler()
        normalized_signal = signal_scaler.fit_transform(filt_signal.reshape(-1, 1)).flatten()
    else: 
        normalized_signal = filt_signal


    return normalized_signal

def get_trainsients(df):
    fz = myfilter(df["Force_Z"], cutoff_freq=30)
    fx = myfilter(df["Force_X"], cutoff_freq=30)

    res_half, res_std, res_rstd = num_transient(fx)
    start_fx = select_index(sliding_sum_window(fx, squared_signal_start=res_half[0]),
                                      sliding_sum_window(fx, squared_signal_start=res_std[0]),
                                      sliding_sum_window(fx, squared_signal_start=res_rstd[0]), fx)
    end_fx = min(start_fx + WS_B, len(fx))
    
    res_half, res_std, res_rstd = num_transient(fz)
    start_fz = select_index(sliding_sum_window(fz, squared_signal_start=res_half[0]),
                                      sliding_sum_window(fz, squared_signal_start=res_std[0]),
                                      sliding_sum_window(fz, squared_signal_start=res_rstd[0]), fz)
    end_fz = min(start_fz + WS_B, len(fz))
    if start_fx < start_fz: return start_fx, end_fx
    else: return start_fz, end_fz

def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        start, end = get_trainsients(df=df)
        if all(col in df.columns for col in ['Force_X', 'Force_Y', 'Force_Z', 'Torque_X', 'Torque_Y', 'Torque_Z', 'Pose_X', 'Pose_Y', 'Pose_Z', 'Y']):
            # Preprocess each force and torque signal
            signals = []
            for col in ['Force_X', 'Force_Y', 'Force_Z', 'Torque_X', 'Torque_Y', 'Torque_Z']:
                signal = preprocess_signal(df[col], start=start, end=end, cutoff_freq=30)
                signals.append(signal)
            
            # Process the delta poses
            pose_columns = ['Pose_X', 'Pose_Y', 'Pose_Z']
            for col in pose_columns:
                delta_pose = np.abs(df[col][0] - df[col])
                delta_pose = preprocess_signal(delta_pose, start=start, end=end, cutoff_freq=15)
                signals.append(delta_pose)

            # Stack the signals along the third axis
            X = np.dstack(signals)            
            y = df.loc[0, "Y"]

            return X, y
        else:
            print(f"Skipping file {csv_path}: Required columns are missing.")
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