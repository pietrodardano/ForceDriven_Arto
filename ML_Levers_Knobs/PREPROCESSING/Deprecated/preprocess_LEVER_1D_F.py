import os
import pandas as pd
import numpy  as np

from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('/home/user/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet, pad_signal_with_noise
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

data_folder = '/home/user/thesis_ws/src/ML_Levers_Knobs/DATA/1D_LEVER_Fx_ScalNorm/'
folder_path = "/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/LEVER/"

def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'Force_X' in df.columns and 'Y' in df.columns:
            Fx = df["Force_X"]
            flag = 0
            filtered = myfilter(Fx, cutoff_freq=30)
            target_length = 2000
            if len(Fx) < target_length:
                padding_length = target_length - len(Fx)
                last_value = Fx.iloc[-1]
                # Pad the signal
                padded_signal = np.pad(Fx, (0, padding_length), mode='constant', constant_values=last_value)
                flag = 1
                noise_mean = 0
                noise_std = np.std(Fx-filtered)
                noise = np.random.normal(noise_mean, noise_std, padding_length)
                padded_signal[-padding_length:] += noise
            elif len(Fx)>target_length:
                Fx=Fx[-target_length:]
            else:
                padded_signal = Fx
                            
            # Apply filtering
            filt_signal = myfilter(padded_signal, cutoff_freq=30)
            print(f"Original: {len(Fx)}, Padded: {len(filt_signal)}, FLAG={flag}")

            # # MEAN NORMALIZATION
            # mean = np.mean(filt_signal)
            # normalized_signal = filt_signal  # / mean

            # NORMALIZATION using StandardScaler
            signal_scaler = StandardScaler()
            normalized_signal = signal_scaler.fit_transform(filt_signal.reshape(-1, 1)).flatten()

            # Get label 'Y'
            y = df.loc[0, "Y"]
            
            return normalized_signal, y
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