import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('/home/rluser/thesis_ws/src/ML/UTILITIES')
from PreProcessingFunctions import myfilter, num_transient, sliding_sum_window, select_index, add_padding, do_wavelet
from PreProcessingFunctions import WS, WS_B
from PreProcessingFunctions import rename_and_convert_to_txt

from sklearn.preprocessing import StandardScaler

from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
import traceback

folder_path = "/home/rluser/thesis_ws/src/RobotData_GRIPA320"
data_folder = "/home/rluser/thesis_ws/src/ML/DATA/HYB_FPCA_GRIP_ScalNorm"

nyq_freq = 0.5 * 500
normalized_cutoff = 30 / nyq_freq
b, a = butter(2, normalized_cutoff, btype='low')


csv_path = '/home/rluser/thesis_ws/src/RobotData_GRIPA320/FCU_ATHR/Butt_7N_1200_#37.csv'


df = pd.read_csv(csv_path)
fz = df["Force_Z"]

signal = myfilter(fz, cutoff_freq=30)
res_half, res_std, res_rstd = num_transient(signal)
start_signal_index = select_index(
    sliding_sum_window(signal, squared_signal_start=res_half[0]),
    sliding_sum_window(signal, squared_signal_start=res_std[0]),
    sliding_sum_window(signal, squared_signal_start=res_rstd[0]),
    signal
)
end_signal_index = min(start_signal_index + WS_B, len(signal))
sliced_signal = signal[start_signal_index:end_signal_index]

# POSE
pXs, pYs, pZs = np.array(df['Pose_X'][start_signal_index:end_signal_index]), \
np.array(df['Pose_Y'][start_signal_index:end_signal_index]), \
np.array(df['Pose_Z'][start_signal_index:end_signal_index])

normalized_cutoff = 15 / nyq_freq
b, a = butter(2, normalized_cutoff, btype='low')
dpos_X, dpos_Y, dpos_Z = np.abs(pXs[0] - pXs), np.abs(pYs[0] - pYs), np.abs(pZs[0] - pZs)
dpos_X, dpos_Y, dpos_Z = filtfilt(b, a, dpos_X), filtfilt(b, a, dpos_Y), filtfilt(b, a, dpos_Z)

# Combine and standardize pose data
pose_data = np.vstack((dpos_X, dpos_Y, dpos_Z)).T
scaler = StandardScaler()
pose_data_standardized = scaler.fit_transform(pose_data)

# Apply PCA
pca = PCA(n_components=3)
pose_pca = pca.fit_transform(pose_data_standardized)
max_variance_index = np.argmax(pca.explained_variance_)
selected_pose_component = pose_pca[:, max_variance_index]

# NORMALIZATION
# if mean > 1 or mean <-1:
#    normalized_signal = sliced_signal  / mean
# else:
#    normalized_signal = sliced_signal

# Not Normalized
# normalized_signal = sliced_signal

# SCALED NORMALIZATION
scaler_Fx = StandardScaler()
normalized_signal = scaler_Fx.fit_transform(sliced_signal.reshape(-1, 1)).flatten()

if np.mean(normalized_signal)<0 : normalized_signal = -normalized_signal

sliced_signal = add_padding(normalized_signal)
selected_pose_component = add_padding(selected_pose_component)
selected_pose_component = selected_pose_component[np.newaxis,:, np.newaxis]

# Apply Continuous Wavelet Transform (CWT)
cwt_matr = do_wavelet(sliced_signal, wavelet='morl')
cwt_matr = cwt_matr.reshape(1, WS_B, len(np.arange(0.25,128))) 
# CWT is RESHAPED to (1, WS_B, scale)

print("CWT shap:", cwt_matr.shape)
print("pse: ", selected_pose_component.shape)

# Stack CWT output and pose component along depth axis
X = np.dstack((cwt_matr, selected_pose_component))  # Depth axis stacking

y = df.loc[0, "Y"]

print(X.shape)
# xc  = input("diocan?")

# print(xc)

X_data = []
X_data.append(X)
X_data.append(X)
X_data.append(X)
#print(X_data.shape)

X_data = np.vstack(X_data)

print(X_data.shape)

##############################################################################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the dimensions of the cube
height = 1465
width = 800
depth = 129

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the vertices of the cube
vertices = [
    (0, 0, 0), (width, 0, 0), (width, depth, 0), (0, depth, 0),
    (0, 0, height), (width, 0, height), (width, depth, height), (0, depth, height)
]

# Define the faces of the cube
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side face 1
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side face 2
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side face 3
    [vertices[3], vertices[0], vertices[4], vertices[7]]   # Side face 4
]

# Define the indices of the faces belonging to the last layer (depth = 129)
last_layer_indices = [2, 3, 4, 5]  # These are the side faces closest to the depth axis

# Plot the cube
for i, face in enumerate(faces):
    # Check if any vertex of the face belongs to the last layer
    if any(v[2] == height for v in face):  # Check if the face is at height (last layer)
        ax.add_collection3d(Poly3DCollection([face], edgecolor='black', linewidths=1, facecolors='red', alpha=0.6))
    else:
        ax.add_collection3d(Poly3DCollection([face], edgecolor='black', linewidths=1, facecolors='cyan', alpha=0.6))

# Set plot labels and limits for height, width, depth
ax.set_xlabel('Width (800)')
ax.set_ylabel('Depth (129)')
ax.set_zlabel('Height (1465)')
ax.set_title("X_data shape")

ax.set_xlim([0, width])
ax.set_ylim([0, depth])
ax.set_zlim([0, height])

# Show plot
plt.show()


