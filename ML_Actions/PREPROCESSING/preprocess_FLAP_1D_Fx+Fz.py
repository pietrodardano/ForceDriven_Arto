import os
import pandas as pd
import numpy  as np

import sys
sys.path.append('/home/user/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet, pad_signal_with_noise
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from sklearn.preprocessing import StandardScaler

data_folder = '/home/user/thesis_ws/src/ML_ACTIONS/DATA/1D_FLAP_Fx+Fz_MeanNorm/'
folder_path = "/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/FLAP/"

def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'Force_X' in df.columns and 'Force_Z' in df.columns and 'Y' in df.columns:
            Fx = df["Force_X"]
            Fz = df["Force_Z"]
            flag = 0
            
            target_length = 1800
            if len(Fx) < target_length:
                padding_length = target_length - len(Fx)
                last_value_Fx = Fx.iloc[-1]
                last_value_Fz = Fz.iloc[-1]
                # Pad the signals
                padded_Fx = np.pad(Fx, (0, padding_length), mode='constant', constant_values=last_value_Fx)
                padded_Fz = np.pad(Fz, (0, padding_length), mode='constant', constant_values=last_value_Fz)
                flag = 1
                noise_mean = 0
                noise_std = 0.1949 # Noise Extracted from #459 from ANN_LT LEVER
                noise = np.random.normal(noise_mean, noise_std, padding_length)
                padded_Fx[-padding_length:] += noise
                padded_Fz[-padding_length:] += noise
            elif len(Fx) > target_length:
                padded_Fx = Fx[-target_length:]
                padded_Fz = Fz[-target_length:]
            else:
                padded_Fx = Fx
                padded_Fz = Fz
            
            # Apply filtering
            filt_Fx = myfilter(padded_Fx, cutoff_freq=30)
            filt_Fz = myfilter(padded_Fz, cutoff_freq=30)
            print(f"Original: {len(Fx)}, Padded: {len(filt_Fx)}, FLAG={flag}")

            # NORMALIZATION (you can adjust this according to your needs)
            mean_Fx = np.mean(filt_Fx)
            mean_Fz = np.mean(filt_Fz)
            if mean_Fx > 1 or mean_Fx <-1:
                normalized_Fx = filt_Fx  / mean_Fx
            else:
                normalized_Fx = filt_Fx
            if mean_Fz > 1 or mean_Fz <-1:
                normalized_Fz = filt_Fz  / mean_Fz
            else:
                normalized_Fz = filt_Fz

            # NOT NORMALIZED
            # normalized_Fx = filt_Fx
            # normalized_Fz = filt_Fz

            # NORMALIZATION using StandardScaler
            # scaler_Fx = StandardScaler()
            # scaler_Fz = StandardScaler()

            # normalized_Fx = scaler_Fx.fit_transform(filt_Fx.reshape(-1, 1)).flatten()
            # normalized_Fz = scaler_Fz.fit_transform(filt_Fz.reshape(-1, 1)).flatten()


            # Combine Force_X and Force_Z
            combined_signal = np.dstack((normalized_Fx, normalized_Fz))

            # Get label 'Y'
            y = df.loc[0, "Y"]
            
            return combined_signal, y
        else:
            print(f"Skipping file {csv_path}: Columns 'Force_X', 'Force_Z', or 'Y' are missing.")
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