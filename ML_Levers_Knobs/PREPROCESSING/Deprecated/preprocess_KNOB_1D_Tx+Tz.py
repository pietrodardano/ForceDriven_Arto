import os
import pandas as pd
import numpy  as np

import sys
sys.path.append('/home/user/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet, pad_signal_with_noise
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from sklearn.preprocessing import StandardScaler

data_folder = '/home/user/thesis_ws/src/ML_Levers_Knobs/DATA/1D_KNOB_Tx+Tz_MeanNorm/'
folder_path = "/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/KNOB_OPEN/"

def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'Torque_X' in df.columns and 'Torque_Z' in df.columns and 'Y' in df.columns:
            Tx = df["Torque_X"]
            Tz = df["Torque_Z"]
            flag = 0

            filt_Tx = myfilter(Tx, cutoff_freq=30)
            filt_Tz = myfilter(Tz, cutoff_freq=30)
            target_length = 2000
            if len(Tx) < target_length:
                padding_length = target_length - len(Tx)
                last_value_Tx = Tx.iloc[-1]
                last_value_Tz = Tz.iloc[-1]
                # Pad the signals
                padded_Tx = np.pad(Tx, (0, padding_length), mode='constant', constant_values=last_value_Tx)
                padded_Tz = np.pad(Tz, (0, padding_length), mode='constant', constant_values=last_value_Tz)
                flag = 1
                noise_mean = 0
                noise_std_x = np.std(Tx-filt_Tx)
                noise_std_z = np.std(Tz-filt_Tz)
                noise_x = np.random.normal(noise_mean, noise_std_x, padding_length)
                noise_z = np.random.normal(noise_mean, noise_std_z, padding_length)
                padded_Tx[-padding_length:] += noise_x
                padded_Tz[-padding_length:] += noise_z
            elif len(Tz)>target_length:
                padded_Tx = Tx[-target_length:]
                padded_Tz = Tz[-target_length:]
            else:
                padded_Tx = Tx
                padded_Tz = Tz
            
            # Apply filtering
            filt_Tx = myfilter(padded_Tx, cutoff_freq=30)
            filt_Tz = myfilter(padded_Tz, cutoff_freq=30)
            print(f"Original: {len(Tx)}, Padded: {len(filt_Tx)}, FLAG={flag}")

            # NORMALIZATION (you can adjust this according to your needs)
            mean_Tx = np.mean(filt_Tx)
            mean_Tz = np.mean(filt_Tz)
            if mean_Tx > 1 or mean_Tx <-1:
                normalized_Tx = filt_Tx  / mean_Tx
            else:
                normalized_Tx = filt_Tx
            if mean_Tz > 1 or mean_Tz <-1:
                normalized_Tz = filt_Tz  / mean_Tz
            else:
                normalized_Tz = filt_Tz

            # NOT NORMALIZED
            # normalized_Tx = filt_Tx
            # normalized_Tz = filt_Tz

            # NORMALIZATION using StandardScaler
            # scaler_Tx = StandardScaler()
            # scaler_Tz = StandardScaler()

            # normalized_Tx = scaler_Tx.fit_transform(filt_Tx.reshape(-1, 1)).flatten()
            # normalized_Tz = scaler_Tz.fit_transform(filt_Tz.reshape(-1, 1)).flatten()


            # Combine Torque_X and Torque_Z
            combined_signal = np.dstack((normalized_Tx, normalized_Tz))

            # Get label 'Y'
            y = df.loc[0, "Y"]
            
            return combined_signal, y
        else:
            print(f"Skipping file {csv_path}: Columns 'Torque_X', 'Torque_Z', or 'Y' are missing.")
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