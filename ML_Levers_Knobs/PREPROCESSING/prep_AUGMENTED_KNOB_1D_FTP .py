import os
import pandas as pd
import numpy  as np
import random

import sys
sys.path.append('/home/rl_sim/TactileDriven_Arto/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet, pad_signal_with_noise
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import butter, filtfilt, resample

data_folder = '/home/rl_sim/TactileDriven_Arto/ML_Levers_Knobs/DATA/1D_AUGM_KNOB_TRANSF_ScalNorm/'
folder_path = "/home/rl_sim/TactileDriven_Arto/ROBOT_ACTIONS_DATA/KNOB/"

target_length = 2000
TONORM = 2

ADD_NOISE = True
TIME_STRETCH = False # CHECK IT
TIME_SHIFT = False  #  BETTER TO NO SHIFT, try it btw

STRETCH_FACTOR = 1.05  #>1 dilatation, <1 compression
SHIFT_MAX = 0 #69
NOISE_FACTOR = 1.05  # +5% noise
AUGMENT_PROB = 0.49  # Probability of augmenting a given data sample


# Functions for augmentation
def add_noise(signal, noise_std):
    noise = np.random.randn(len(signal))*noise_std*NOISE_FACTOR,
    augmented_signal = signal + noise
    return augmented_signal

def time_shift(signal, shift_max=SHIFT_MAX):   # CAN CUSE PROBLEM CREATING FALSE SLOPES !!
    shift = np.random.randint(-shift_max, shift_max)
    augmented_signal = np.roll(signal, shift)
    return augmented_signal

# THIS ONE SUPPOSES THAT SIGNALS ARE PERIODIC, MINE ARE NOT !!!!!
# def time_stretch(signal, stretch_factor=STRETCH_FACTOR):
#     length = int(len(signal) / stretch_factor)
#     augmented_signal = resample(signal, length)
#     return augmented_signal

def time_stretch(signal, stretch_factor=STRETCH_FACTOR):
    x = np.arange(len(signal))
    new_length = int(len(signal) * stretch_factor)
    x_new = np.linspace(0, len(signal) - 1, new_length)
    augmented_signal = np.interp(x_new, x, signal)
    return augmented_signal

def apply_augmentation(signal, augmentations):
    augmented_signal = signal
    for aug in augmentations:
        augmented_signal = aug(augmented_signal)
    return augmented_signal

def preprocess_signal(signal, cutoff_freq=30, target_length=target_length, tonorm=TONORM):
    filtered_signal = myfilter(signal, cutoff_freq)
    if len(signal) < target_length:
        padding_length = target_length - len(signal)
        last_value = signal.iloc[-1] if isinstance(signal, pd.Series) else signal[-1]
        # Pad the signal
        padded_signal = np.pad(signal, (0, padding_length), mode='constant', constant_values=last_value)
        noise_mean = 0
        noise_std = np.std(signal - filtered_signal)
        noise = np.random.normal(noise_mean, noise_std, padding_length)
        padded_signal[-padding_length:] += noise
    elif len(signal) > target_length:
        padded_signal = signal[-target_length:]
    else:
        padded_signal = signal

    filt_signal = myfilter(padded_signal, cutoff_freq)
    

    if tonorm == 2:
        # NORMALIZATION using StandardScaler
        signal_scaler = StandardScaler()
        normalized_signal = signal_scaler.fit_transform(filt_signal.reshape(-1, 1)).flatten()
    elif tonorm == 1:
        # NORMALIZATION using MinMaxScaler
        signal_scaler = MinMaxScaler()
        normalized_signal = signal_scaler.fit_transform(filt_signal.reshape(-1, 1)).flatten()
    else: 
        normalized_signal = filt_signal

    return normalized_signal, np.std(signal-filtered_signal)

def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if all(col in df.columns for col in ['Force_X', 'Force_Y', 'Force_Z', 'Torque_X', 'Torque_Y', 'Torque_Z', 'Pose_X', 'Pose_Y', 'Pose_Z', 'Y']):
            # Preprocess each force and torque signal
            signals = []
            augmented_signals_list = []

            # Determine if the whole set should be augmented
            augment_whole_set = random.random() < AUGMENT_PROB

            for col in ['Force_X', 'Force_Y', 'Force_Z', 'Torque_X', 'Torque_Y', 'Torque_Z']:
                signal = df[col]

                # Create a duplicate for augmentation
                aug_signal = signal.copy()

                # Apply time shift and time stretch augmentations before preprocessing if needed
                if augment_whole_set:
                    if TIME_SHIFT and random.random() < 0.5:
                        aug_signal = time_shift(aug_signal)
                    if TIME_STRETCH and random.random() < 0.5:
                        aug_signal = time_stretch(aug_signal)
                        # Ensure the length is consistent for both original and augmented signals
                        if len(aug_signal) != target_length:
                            aug_signal = resample(aug_signal, target_length)

                # Preprocess the signals
                processed_signal, _ = preprocess_signal(signal, cutoff_freq=30, target_length=target_length)
                signals.append(processed_signal)

                if augment_whole_set:
                    # Preprocess the duplicated signal
                    aug_signal, noise_std = preprocess_signal(aug_signal, cutoff_freq=30, target_length=target_length)
                    # Apply noise augmentation after preprocessing
                    if ADD_NOISE and random.random() < 0.5:
                        aug_signal = add_noise(aug_signal, noise_std)
                    augmented_signals_list.append(aug_signal)

            # Process the delta poses
            pose_columns = ['Pose_X', 'Pose_Y', 'Pose_Z']
            for col in pose_columns:
                delta_pose = np.abs(df[col][0] - df[col])

                # Create a duplicate for augmentation
                aug_delta_pose = delta_pose.copy()

                # Apply time shift and time stretch augmentations before preprocessing if needed
                if augment_whole_set:
                    if TIME_SHIFT and random.random() < 0.5:
                        aug_delta_pose = time_shift(aug_delta_pose)
                    if TIME_STRETCH and random.random() < 0.5:
                        aug_delta_pose = time_stretch(aug_delta_pose)
                        # Ensure the length is consistent for both original and augmented signals
                        if len(aug_delta_pose) != target_length:
                            aug_delta_pose = resample(aug_delta_pose, target_length)

                # Preprocess the delta poses
                processed_delta_pose, _ = preprocess_signal(delta_pose, cutoff_freq=15, target_length=target_length)
                signals.append(processed_delta_pose)

                if augment_whole_set:
                    # Preprocess the duplicated delta pose
                    aug_delta_pose, noise_std= preprocess_signal(aug_delta_pose, cutoff_freq=15, target_length=target_length)
                    # Apply noise augmentation after preprocessing
                    if ADD_NOISE and random.random() < 0.5:
                        aug_delta_pose = add_noise(aug_delta_pose, noise_std)
                    augmented_signals_list.append(aug_delta_pose)

            X = np.dstack(signals)
            y = df.loc[0, "Y"]

            if augment_whole_set:
                augmented_X = np.dstack(augmented_signals_list)
                return [X, augmented_X], y
            else:
                return [X], y
        else:
            print(f"Skipping file {csv_path}: Required columns are missing.")
            rename_and_convert_to_txt(csv_path)
            return None, None
    except Exception as e:
        print(f"Error processing file {csv_path}: {e}")
        return None, None

def preprocess_folder_data(folder_path, data_folder):
    os.makedirs(data_folder, exist_ok=True)
    c = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                X_data_list, y_data = preprocess_data(file_path)
                if X_data_list is not None and y_data is not None:
                    for X_data in X_data_list:
                        file_name = os.path.splitext(file)[0] + f"#{c}_preprocessed.npy"
                        save_path = os.path.join(data_folder, file_name)
                        np.savez(save_path, X=X_data, y=y_data)
                        print(f"Preprocessed data saved to {save_path}")
                        c += 1

# Call the preprocess_folder_data function
preprocess_folder_data(folder_path, data_folder)