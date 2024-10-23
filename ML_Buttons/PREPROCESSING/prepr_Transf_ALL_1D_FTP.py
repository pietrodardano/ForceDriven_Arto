import os
import pandas as pd
import numpy  as np

import sys
sys.path.append('/home/rluser/TactileDriven_Arto/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet, pad_signal_with_noise
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import butter, filtfilt

data_folder = '/home/rluser/TactileDriven_Arto/ML_Buttons/DATA/1D_TRANSF_FTP_NotNorm/'
folder_path = "/home/rluser/TactileDriven_Arto/ROBOT_ACTIONS_DATA/BUTTONS"
from scipy.spatial.transform import Rotation as R


def transform_force_torque_to_tcp(tcp_pose, force_base):
    rotation_vector = np.array(tcp_pose[3:])
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    force_base_vector = np.array(force_base[:3])
    torque_base_vector = np.array(force_base[3:])
    force_tcp_vector = np.dot(rotation_matrix.T, force_base_vector)
    torque_tcp_vector = np.dot(rotation_matrix.T, torque_base_vector)
    
    return np.concatenate((force_tcp_vector, torque_tcp_vector))

def preprocess_signal(signal, cutoff_freq=30, start=0, end=2000, tonorm=0):
    signal = myfilter(signal, cutoff_freq)
    sliced_signal = signal[start: end]
    if end - start != WS_B:
        print("!! -- ERROR -- !! end-start not WS_B")

    filt_signal = myfilter(sliced_signal, cutoff_freq)

    if tonorm == 1:
        signal_scaler = MinMaxScaler()
        normalized_signal = signal_scaler.fit_transform(filt_signal.reshape(-1, 1)).flatten()
    elif tonorm == 2:
        signal_scaler = StandardScaler()
        normalized_signal = signal_scaler.fit_transform(filt_signal.reshape(-1, 1)).flatten()  # Standard scaling
    else:
        normalized_signal = filt_signal

    return normalized_signal

def calculate_energy(signal, window_size):
    return np.array([np.sum(signal[i:i + window_size] ** 2) for i in range(len(signal) - WS_B + 1)])

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

def preprocess_data(data):
    try:
        # Check if necessary columns are present
        required_columns = ['Force_X', 'Force_Y', 'Force_Z', 'Torque_X', 'Torque_Y', 'Torque_Z', 'Pose_X', 'Pose_Y', 'Pose_Z', 'Pose_Rx', 'Pose_Ry', 'Pose_Rz', 'Y']
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
        
        # Get transient start and end indices using the energy-based detector on Force_Z_TCP
        start, end = get_transient_start_index(transformed_signals[:, 2])

        # Preprocess each transformed signal
        signals = []
        for col_idx in range(6):  # 6 columns: Force_X_TCP, Force_Y_TCP, Force_Z_TCP, Torque_X_TCP, Torque_Y_TCP, Torque_Z_TCP
            signal = preprocess_signal(transformed_signals[:, col_idx], start=start, end=end, cutoff_freq=30)
            signals.append(signal)

        # Process the delta poses
        delta_poses = []
        for col in ['Pose_X', 'Pose_Y', 'Pose_Z']:
            delta_pose = np.abs(data[col].iloc[0] - data[col])
            delta_pose = preprocess_signal(delta_pose.to_numpy(), start=start, end=end, cutoff_freq=15)
            delta_poses.append(delta_pose)

        # Stack the signals along the third axis
        X = np.dstack(signals + delta_poses)

        return X, y
    except Exception as e:
        print(f"Error processing data: {e}")
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