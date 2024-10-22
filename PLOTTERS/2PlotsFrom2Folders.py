import os
import random
import pandas as pd
import matplotlib.pyplot as plt

folder1 = "/home/user/thesis_ws/src/RobotData_SIMA320"
folder2 = "/home/user/thesis_ws/src/RobotData_GRIPA320"

# Get list of subfolders in both folders
# subfolders1 = os.listdir(folder1)
# subfolders2 = os.listdir(folder2)

subfolders1 = subfolders2 = ['MCDU_5', 'TERR_but','MCDU_INIT', 'VH_greenArrow', 'MCDU_L', 'MCDU_CLR', 'AUTOBRAKE', 'ECAM_FCTL']


# Sort subfolders to ensure consistency
subfolders1.sort()
subfolders2.sort()
c=0
# Iterate over subfolders and plot corresponding files
for subfolder1, subfolder2 in zip(subfolders1, subfolders2):
    files1 = os.listdir(os.path.join(folder1, subfolder1))
    files2 = os.listdir(os.path.join(folder2, subfolder2))

    # Sort files to ensure consistency
    files1.sort()
    files2.sort()

    # Select up to 6 random similar files
    similar_files = random.sample(list(zip(files1, files2)), min(18, min(len(files1), len(files2))))

    for file1, file2 in similar_files:
        if file1.split('_')[:-1] == file2.split('_')[:-1]:
            # Load data from CSV files using pandas
            df1 = pd.read_csv(os.path.join(folder1, subfolder1, file1))
            df2 = pd.read_csv(os.path.join(folder2, subfolder2, file2))

            # Extract 'Force_Z' column data
            force_z1 = df1['Force_Z']
            y1 = df1.loc[0, 'Y']
            y2 = df2.loc[0, 'Y']
            force_z2 = df2['Force_Z']

            # Plotting
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.plot(force_z1, label='F_Z')
            plt.title('F1- ' + subfolder1 + ' - ' + file1 + f'Y {y1}')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(force_z2, label='F_Z')
            plt.title('F2- ' + subfolder2 + ' - ' + file2+ f'Y {y2}')
            plt.legend()

            plt.show()


