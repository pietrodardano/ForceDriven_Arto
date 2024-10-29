import os
import pandas as pd
import numpy  as np

import sys
sys.path.append('/home/rl_sim/TactileDriven_Arto/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet, pad_signal_with_noise
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

data_folder = '/home/rl_sim/TactileDriven_Arto/ML_Levers_Knobs/DATA/1D_TRANSF_LEVER_FTP_ScalNorm/'
folder_path = "/home/rl_sim/TactileDriven_Arto/ROBOT_ACTIONS_DATA/LEVER/"

target_length = 2000
"""
LEVER --> 2000
KNOBS --> 2000
"""

def preprocess_signal(signal, cutoff_freq=30, target_length=target_length, tonorm=2):
    """
    Preprocesses a signal by filtering, padding/truncating, and normalizing.
    """
    filtered_signal = myfilter(signal, cutoff_freq)

    if len(signal) < target_length:
        # Pad the signal with its last value plus some noise
        padding_length = target_length - len(signal)
        last_value = signal.iloc[-1] if isinstance(signal, pd.Series) else signal[-1]
        padded_signal = np.pad(signal, (0, padding_length), mode='constant', constant_values=last_value)
        
        noise_mean = 0
        noise_std = np.std(signal - filtered_signal)
        noise = np.random.normal(noise_mean, noise_std, padding_length)
        padded_signal[-padding_length:] += noise
    elif len(signal) > target_length:
        # Truncate the signal
        padded_signal = signal[-target_length:]
    else:
        # No padding/truncating needed
        padded_signal = signal

    # Apply filtering
    filt_signal = myfilter(padded_signal, cutoff_freq)

    # Normalize the signal
    mean = np.mean(filt_signal)
    if tonorm == 1 and mean != 0:
        normalized_signal = filt_signal / mean  # Mean normalization
    elif tonorm == 2:
        signal_scaler = StandardScaler()
        normalized_signal = signal_scaler.fit_transform(filt_signal.reshape(-1, 1)).flatten()  # Standard scaling
    elif tonorm == 3:
        signal_scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_signal = signal_scaler.fit_transform(filt_signal.reshape(-1, 1)).flatten()
    else:
        normalized_signal = filt_signal

    return normalized_signal

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
    Processes a DataFrame by transforming and normalizing the force/torque data.
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
        for col_idx in range(3):  # 6 columns: Force_X_TCP, Force_Y_TCP, Force_Z_TCP, Torque_X_TCP, Torque_Y_TCP, Torque_Z_TCP
            signal = preprocess_signal(transformed_signals[:, col_idx], cutoff_freq=30)
            signals.append(signal)
        for col_idx in range(3,6):  # 6 columns: Force_X_TCP, Force_Y_TCP, Force_Z_TCP, Torque_X_TCP, Torque_Y_TCP, Torque_Z_TCP
            signal = preprocess_signal(transformed_signals[:, col_idx], cutoff_freq=30)
            signals.append(signal)

        # Process the delta poses
        delta_poses = []
        for col in ['Pose_X', 'Pose_Y', 'Pose_Z']:
            delta_pose = np.abs(data[col].iloc[0] - data[col])
            delta_pose = preprocess_signal(delta_pose.to_numpy(), cutoff_freq=15)
            delta_poses.append(delta_pose)

        # Stack the signals along the third axis
        X = np.dstack(signals + delta_poses)

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