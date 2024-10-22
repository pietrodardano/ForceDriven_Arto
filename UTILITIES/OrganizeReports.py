import os
import inspect
from sklearn.metrics import accuracy_score, f1_score, classification_report
#import re
import numpy as np

def find_matching_file_path(model_name, building_function, history, y_test, y_pred_binary, classification_rep_current):
    counter = 1
    while os.path.exists(f"MODELS_SUMMARIES/{model_name}/BuildFcn#{counter}.txt"):
        with open(f"MODELS_SUMMARIES/{model_name}/BuildFcn#{counter}.txt", "r") as file:
            lines = file.readlines()
        
        # Extract the building function code from the file
        start_idx = lines.index(inspect.getsource(building_function) + '\n')

        if start_idx >= 0:
            return f"MODELS_SUMMARIES/{model_name}/BuildFcn#{counter}.txt"
        
        counter += 1
    FLAG = 0
    # If no matching file exists, create a new one
    file_path = f"MODELS_SUMMARIES/{model_name}/BuildFcn#{counter}.txt"
    with open(file_path, "w") as file:
        file.write(inspect.getsource(building_function) + '\n')
        FLAG = 1
        # Create the model to display its summary
        model = building_function()
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))

        #NO PRINTS, just calculatuons
        accuracy = accuracy_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        # Write the model summary to the file
        file.write("\n")
        file.write("Model Summary:\n")
        for line in model_summary:
            file.write(line + '\n')
        
        file.write("\n##### WORST #####\n")
        file.write("Train loss: 0.01\nTest val_loss: 0\nTrain accuracy: 0.99\nAccuracy Score: 0.99 \nF1 Score: 0.99")
        file.write("\n")
        file.write("##### BEST #####\n")
        file.write(f"Train loss: {history['loss'][-1]} \n")
        file.write(f"Test val_loss: {history['val_loss'][-1]} \n")
        file.write(f"Train accuracy: {history['accuracy'][-1]} \n")
        file.write(f"Accuracy Score: {accuracy} \n")
        file.write(f"F1 Score: {f1} \n")
        file.write(f"Classification Report:\n {classification_rep_current}\n\n")
        file.write("\n")
    
    return file_path, FLAG

def update_performance_in_file(file_path, history, y_test, y_pred_binary, classification_rep_current):
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract the worst and best performance metrics
    worst_section_start = lines.index("##### WORST #####\n") + 1
    worst_section_end = lines.index("##### BEST #####\n")
    worst_lines = lines[worst_section_start: worst_section_end]

    best_section_start = worst_section_end + 1
    best_lines = lines[best_section_start:]

    # Extract the worst and best performance metrics
    worst_metrics = {}
    best_metrics = {}
    worst_classification_rep = ""
    best_classification_rep = ""

    for line in worst_lines:
        if "Classification Report:" in line:
            worst_classification_rep = next(file).strip()
        else:
            key, value = line.strip().split(": ")
            worst_metrics[key] = float(value)

    for line in best_lines:
        if "Classification Report:" in line:
            best_classification_rep = next(file).strip()
        else:
            key, value = line.strip().split(": ")
            best_metrics[key] = float(value)

    # Calculate new performance metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    # Update the worst and best performance metrics and classification reports if necessary
    updated_w, updated_b = False, False

    if history['loss'] > worst_metrics['Train loss'] or accuracy < worst_metrics['Accuracy Score']:
        worst_metrics['Train loss'] = history['loss']
        worst_metrics['Test val_loss'] = history['val_loss']
        worst_metrics['Train accuracy'] = history['accuracy']
        worst_metrics['Accuracy Score'] = accuracy
        worst_metrics['F1 Score'] = f1
        worst_classification_rep = classification_rep_current
        updated_w = True

    if history['loss'] < best_metrics['Train loss'] or accuracy > best_metrics['Accuracy Score']:
        best_metrics['Train loss'] = history['loss']
        best_metrics['Test val_loss'] = history['val_loss']
        best_metrics['Train accuracy'] = history['accuracy']
        best_metrics['Accuracy Score'] = accuracy
        best_metrics['F1 Score'] = f1
        best_classification_rep = classification_rep_current
        updated_b = True

    # Write the updated worst and best performance metrics and classification reports back to the file
    if updated_w or updated_b:
        with open(file_path, 'w') as file:
            # Rewrite the worst metrics
            file.writelines(lines[:worst_section_start])
            
            file.write("##### WORST #####\n")
            for key, value in worst_metrics.items():
                file.write(f"{key}: {value}\n")
            file.write(f"Classification Report:\n {worst_classification_rep}\n\n")

            file.write("##### BEST #####\n")
            for key, value in best_metrics.items():
                file.write(f"{key}: {value}\n")
            file.write(f"Classification Report:\n {best_classification_rep}\n\n")
            
            # # Rewrite the rest of the lines
            # file.writelines(lines[best_section_start:])

    return updated_w, updated_b

def compare_and_organize(model_name, building_function, history, y_test, y_pred_binary):
    file_path, FLAG = find_matching_file_path(model_name, building_function, history, y_test, y_pred_binary, classification_report(y_test, y_pred_binary))
    if FLAG==0: 
        updated_w, updated_b = update_performance_in_file(file_path, history, y_test, y_pred_binary, classification_report(y_test, y_pred_binary))
        if updated_w:
            print("Updated WORST model Report")
        elif updated_b:
            print("Updated BEST model Report")
        else:
            print("NO UPDATE REPORTS")

