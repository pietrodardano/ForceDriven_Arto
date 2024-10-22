import os
import pandas as pd
import numpy  as np

import sys
sys.path.append('/home/rluser/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from sklearn.preprocessing import StandardScaler

from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
import traceback

folder_path = "/home/rluser/thesis_ws/src/RobotData_GRIPA320"
data_folder = "/home/rluser/thesis_ws/src/ML/DATA/1D_FPCA_GRIP_ScalNorm"

nyq_freq = 0.5 * 500
normalized_cutoff = 30 / nyq_freq
b, a = butter(2, normalized_cutoff, btype='low')
    
def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'Force_Z' in df.columns and 'Y' in df.columns:
            fz = df["Force_Z"]

            signal = myfilter(fz, cutoff_freq=30)
            res_half, res_std, res_rstd = num_transient(signal)
            start_signal_index = select_index(sliding_sum_window(signal, squared_signal_start=res_half[0]),
                                              sliding_sum_window(signal, squared_signal_start=res_std[0]),
                                              sliding_sum_window(signal, squared_signal_start=res_rstd[0]), signal)
            end_signal_index = min(start_signal_index + WS_B, len(signal))
            sliced_signal = signal[start_signal_index: end_signal_index]

            pXs, pYs, pZs = np.array(df['Pose_X'][start_signal_index: end_signal_index]), \
                            np.array(df['Pose_Y'][start_signal_index: end_signal_index]), \
                            np.array(df['Pose_Z'][start_signal_index: end_signal_index])
            
            normalized_cutoff = 15 / nyq_freq
            b, a = butter(2, normalized_cutoff, btype='low')
            dpos_X, dpos_Y, dpos_Z = np.abs(pXs[0] - pXs), np.abs(pYs[0] - pYs), np.abs(pZs[0] - pZs)
            dpos_X, dpos_Y, dpos_Z = filtfilt(b, a, dpos_X), filtfilt(b, a, dpos_Y), filtfilt(b, a, dpos_Z)

            # Combine pose data
            pose_data = np.vstack((dpos_X, dpos_Y, dpos_Z)).T
            
            # Standardize the pose data
            scaler = StandardScaler()
            pose_data_standardized = scaler.fit_transform(pose_data)

            # Apply PCA
            pca = PCA(n_components=3)
            pose_pca = pca.fit_transform(pose_data_standardized)

            # Find the component with the highest variance
            max_variance_index = np.argmax(pca.explained_variance_)
            selected_pose_component = pose_pca[:, max_variance_index]

            # NORMALIZATION
            # if mean > 1 or mean <-1:
            #     normalized_signal = sliced_signal  / mean
            # else:
            #     normalized_signal = sliced_signal

            # Not Normalized
            normalized_signal = sliced_signal

            # SCALED NORMALIZATION
            #scaler_Fx = StandardScaler()
            #normalized_signal = scaler_Fx.fit_transform(sliced_signal.reshape(-1, 1)).flatten()

            if np.mean(normalized_signal)<0 : normalized_signal = -normalized_signal

            sliced_signal = add_padding(normalized_signal)
            selected_pose_component = add_padding(selected_pose_component)

            X = np.dstack((sliced_signal, selected_pose_component))

            y = df.loc[0, "Y"]
            return X, y
        else:
            print(f"Skipping file {csv_path}: Columns 'Force_Z' or 'Y' are missing.")
            rename_and_convert_to_txt(csv_path)
            return None, None
    except Exception as e:
        print(f"Exception: Error processing file {csv_path}: {e}")
        traceback.print_exc()
        return None, None

def preprocess_folder_data(folder_path, data_folder):
    # Create the data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    c = 0
    
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
                    c += 1
    return None

preprocess_folder_data(folder_path, data_folder)