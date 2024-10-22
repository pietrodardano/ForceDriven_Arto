import os
import pandas as pd

def modifyYinCSV(root_folder):
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                
                if 'Y' in df.columns and 'Correct' in df.columns:
                    y_value = df.at[0, 'Y']
                    correct_value = df.at[0, 'Correct']

                    if y_value == 1:
                        if correct_value == 0:
                            df.at[0, 'Y'] = 5
                    elif y_value == 2:
                        if correct_value == 0:
                            df.at[0, 'Y'] = 4
                    elif y_value == 3:
                        if correct_value == 0:
                            df.at[0, 'Y'] = 6

                    # Drop the 'Correct' column
                    df.drop(columns=['Correct'], inplace=True)

                    # Save the updated CSV file
                    df.to_csv(file_path, index=False)
                    print(f"Processed file: {file_path}")

# Get the folder path from user input
folder_path = input("Enter the path to the folder: ")

# Process the CSV files in the specified folder
modifyYinCSV(folder_path)
