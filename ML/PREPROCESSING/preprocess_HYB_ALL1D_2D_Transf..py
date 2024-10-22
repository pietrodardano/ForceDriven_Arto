import os
import pandas as pd
import numpy  as np

import sys
sys.path.append('/home/rl_sim/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, add_padding, do_wavelet, pad_signal_with_noise
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt
import pywt

from sklearn.preprocessing import StandardScaler

from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
import traceback

folder_path = "/home/rl_sim/thesis_ws/src/RobotData_GRIPA320"
data_folder = "/home/rl_sim/thesis_ws/src/ML/DATA/HYB_1FTP_2FT_NotNorm"

nyq_freq = 0.5 * 500
normalized_cutoff = 30 / nyq_freq
b, a = butter(2, normalized_cutoff, btype='low')

target_length = 800

# def do_wavelet(signal, wavelet='morl', scales=np.arange(0.25, 128)):
#     cwt_matr, _ = pywt.cwt(signal, scales, wavelet)
#     return cwt_matr


def preprocess_signal(signal, cutoff_freq=30, start=0, end=800, tonorm=0):
    signal = myfilter(signal, cutoff_freq)
    sliced_signal = signal[start: end]
    if end - start != WS_B:
        print("!! -- ERROR -- !! end-start not ws_b")

    filt_signal = myfilter(sliced_signal, cutoff_freq)

    mean = np.mean(filt_signal)
    if tonorm == 1 and mean != 0:     
        #  MEAN NORMALIZATION
        normalized_signal = filt_signal / mean
    elif tonorm == 2:
        # NORMALIZATION using StandardScaler
        signal_scaler = StandardScaler()
        normalized_signal = signal_scaler.fit_transform(filt_signal.reshape(-1, 1)).flatten()
    else: 
        normalized_signal = filt_signal

    return normalized_signal

def calculate_energy(signal, window_size):
    return np.array([np.sum(signal[i:i + window_size] ** 2) for i in range(len(signal) - window_size + 1)])

def find_threshold_indices(energy, max_energy, thresholds):
    indices = []
    for threshold in thresholds:
        for i, e in enumerate(energy):
            if e >= max_energy * threshold:
                indices.append(i)
                break
    return indices

def get_transient_start_index(signal):

    filtered_signal = myfilter(signal, cutoff_freq=30)
    window_size = 300
    energy = calculate_energy(filtered_signal, window_size)
    max_energy = np.max(energy)

    thresholds = [0.38, 0.28, 0.22]

    indices = find_threshold_indices(energy, max_energy, thresholds)
    if abs(indices[1] - indices[0]) <= 60 and abs(indices[2] - indices[1]) <= 60:
        start_index = indices[1]
    elif abs(indices[1] - indices[0]) <= 100 and abs(indices[2] - indices[1]) <= 100:
        start_index = indices[0]
    elif abs(indices[2] - indices[0]) > 250:
        start_index = indices[2]
    else:
        start_index = indices[1]  # Default to the middle index if none of the conditions are met
    
    start_index = max(0, start_index)
    end_index = min(start_index + WS_B, len(signal))
    if end_index == len(signal): 
        start_index = end_index-WS_B

    return start_index, end_index

def transform_force_torque_to_tcp(tcp_pose, force_base):
    """
    Transforms force and torque data from the base frame to the TCP frame.
    """
    rotation_vector = np.array(tcp_pose[3:])
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    force_base_vector = np.array(force_base[:3])
    torque_base_vector = np.array(force_base[3:])
    force_tcp_vector = np.dot(rotation_matrix.T, force_base_vector)
    torque_tcp_vector = np.dot(rotation_matrix.T, torque_base_vector)
    
    return np.concatenate((force_tcp_vector, torque_tcp_vector))

def preprocess_data(data):
    """
    Processes a DataFrame by transforming and normalizing the force/torque data and performing wavelet transformation.
    """
    try:
        # Check if necessary columns are present
        required_columns = ['Force_X', 'Force_Y', 'Force_Z', 'Torque_X', 'Torque_Y', 'Torque_Z', 
                            'Pose_X', 'Pose_Y', 'Pose_Z', 'Pose_Rx', 'Pose_Ry', 'Pose_Rz', 'Y']
        if not all(col in data.columns for col in required_columns):
            print(f"Skipping data: Required columns are missing.")
            return None, None

        # Convert data to NumPy arrays for faster processing
        tcp_pose = data[['Pose_X', 'Pose_Y', 'Pose_Z', 'Pose_Rx', 'Pose_Ry', 'Pose_Rz']].to_numpy()
        forces = data[['Force_X', 'Force_Y', 'Force_Z']].to_numpy()
        torques = data[['Torque_X', 'Torque_Y', 'Torque_Z']].to_numpy()
        y = data['Y'].values[0]

        # Transform forces and torques to the TCP frame before preprocessing
        transformed_signals = []
        for i in range(len(data)):
            transformed_signal = transform_force_torque_to_tcp(tcp_pose[i], np.concatenate((forces[i], torques[i])))
            transformed_signal[2] = -transformed_signal[2]  # Invert the sign of Force_Z_TCP after the transformation
            transformed_signals.append(transformed_signal)
        
        transformed_signals = np.array(transformed_signals)

        # Preprocess each transformed signal
        signals = []
        cwt_signals = []
        for col_idx in range(6):  # 6 columns: Force_X_TCP, Force_Y_TCP, Force_Z_TCP, Torque_X_TCP, Torque_Y_TCP, Torque_Z_TCP
            signal = preprocess_signal(transformed_signals[:, col_idx], cutoff_freq=30)
            signals.append(signal)

            # Perform CWT and reshape
            cwt_signal, _ = pywt.cwt(data=signal, scales=np.arange(0.25, 128), wavelet='morl')
            cwt_signal = cwt_signal.reshape(1, target_length, len(np.arange(0.25, 128)))
            cwt_signals.append(cwt_signal)

        # Process the delta poses
        delta_poses = []
        for col in ['Pose_X', 'Pose_Y', 'Pose_Z']:
            delta_pose = np.abs(data[col].iloc[0] - data[col])
            delta_pose = preprocess_signal(delta_pose.to_numpy(), cutoff_freq=15)
            delta_poses.append(delta_pose)

        # Stack the signals along the third axis
        X_1D = np.dstack(signals + delta_poses)
        X_2D = np.dstack(cwt_signals)

        # Combine 1D and 2D data
        X = np.concatenate([X_1D, X_2D], axis=2)

        return X, y
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

def preprocess_folder_data(folder_path, data_folder):
    """
    Processes all CSV files in a folder, saving the processed data as .npy files.
    """
    os.makedirs(data_folder, exist_ok=True)
    c = 0
    
    # Traverse the folder structure
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)
                X_data, y_data = preprocess_data(data)
                if X_data is not None and y_data is not None:
                    # Save preprocessed data into a file in the data folder
                    file_name = os.path.splitext(file)[0] + f"#{c}_preprocessed.npy"
                    save_path = os.path.join(data_folder, file_name)
                    np.savez(save_path, X=X_data, y=y_data)
                    print(f"Preprocessed data saved to {save_path}")
                    c += 1


preprocess_folder_data(folder_path, data_folder)