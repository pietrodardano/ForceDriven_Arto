import os
import pandas as pd

def count_values_in_y(folder_path):
    count_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4':0, '5':0, '6':0}

    def process_csv(file_path):
        nonlocal count_dict
        try:
            df = pd.read_csv(file_path, nrows=1)  # Read only the first row
            if 'Y' in df.columns:
                y_value = str(int(df['Y'].iloc[0]))  # Convert to string for dictionary key
                #print("N read: ", y_value)
                if y_value in count_dict:
                    count_dict[y_value] += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    def print_first_row(csv_file_path):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            
            # Print the first row
            print("First row of the CSV file:")
            print(df.iloc[0])
        except Exception as e:
            print(f"Error reading or printing the first row of {csv_file_path}: {e}")


    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                #print_first_row(os.path.join(root, file))
                process_csv(os.path.join(root, file))

    return count_dict

# Folder path
folder_path_ = ['/home/rluser/thesis_ws/src/RobotData_GRIPA320','/home/rluser/thesis_ws/src/ROBOT_ACTIONS_DATA/LDG/', '/home/rluser/thesis_ws/src/ROBOT_ACTIONS_DATA/SPEED/', '/home/rluser/thesis_ws/src/ROBOT_ACTIONS_DATA/KNOB_OPEN/']

# Count occurrences of values in the first element of column "Y"
for folder_path in folder_path_:
    count_dict = count_values_in_y(folder_path)
    print(f"\n-------- FOLDER: {folder_path[44:]} -------")
    print("Occurrences of values in the first element of column 'Y':")
    tot=0
    for value, count in count_dict.items():
        print(f"Value {value}: {count} occurrences")
        tot += count
    print(f"+++ TOTAL DATA: {tot} +++")
