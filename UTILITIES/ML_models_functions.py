import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import inspect

import tensorflow as tf

def free_gpu_memory():
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for each GPU
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=(3.8)*1024)])  # Memory limit in MB
        except RuntimeError as e:
            print(e)

def save_summary_and_results(model, history, loss, accuracy, f1, model_name, y_test, y_pred_binary, building_function):
    folder_name = "MODELS_SUMMARIES/" + model_name
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, "dscrpt.txt")

    with open(file_path, "w") as file:                  # CHANGE TO "a"  ## and skip "is_better_performance" function
        # Save building function
        file.write(inspect.getsource(building_function) + '\n')
        # Save model summary
        model.summary(print_fn=lambda x: file.write(x + '\n'))

        file.write("Model Configuration:\n")
        file.write(f"Optimizer: {model.optimizer}\n")
        file.write(f"Loss Function: {model.loss}\n")
        file.write(f"Learning Rate: {model.optimizer.learning_rate}\n\n")
        
        # Save training and testing results
        file.write(f"Train loss: {history.history['loss'][-1]}\n")
        file.write(f"Test val_loss: {history.history['val_loss'][-1]}\n")
        file.write(f"Train accuracy: {history.history['accuracy'][-1]}\n")
        file.write(f"Accuracy Score: {accuracy}\n")
        file.write(f"F1 Score: {f1}\n")
        file.write(f"Classification Report:\n {classification_report(y_test, y_pred_binary)}\n")
        
        # Save complete training history
        file.write("Training History:\n")
        for key, value in history.history.items():
            file.write(f"{key}: {value}\n")

        file.write("\nConfusion Matrix:\n")
        conf_mat = confusion_matrix(y_test, y_pred_binary)
        file.write(np.array2string(conf_mat) + '\n')

        file.write("\n################################################################################################ \n\n")

def save_datasummary_and_results(model, history, loss, accuracy, f1, model_name, y_test, y_pred_binary, building_function, assign_and_deploy_function):
    folder_name = "MODELS_SUMMARIES/" + model_name
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, "dscrpt.txt")

    with open(file_path, "w") as file:  # CHANGE TO "a" if you want to append instead of overwrite
        # Save building function
        file.write("Building Function:\n")
        file.write(inspect.getsource(building_function) + '\n\n')
        
        # Save assign_and_deploy_variables function
        file.write("Assign and Deploy Variables Function:\n")
        file.write(inspect.getsource(assign_and_deploy_function) + '\n\n')

        # Save model summary
        model.summary(print_fn=lambda x: file.write(x + '\n'))

        file.write("Model Configuration:\n")
        file.write(f"Optimizer: {model.optimizer}\n")
        file.write(f"Loss Function: {model.loss}\n")
        file.write(f"Learning Rate: {model.optimizer.learning_rate}\n\n")
        
        # Save training and testing results
        file.write(f"Train loss: {history.history['loss'][-1]}\n")
        file.write(f"Test val_loss: {history.history['val_loss'][-1]}\n")
        file.write(f"Train accuracy: {history.history['accuracy'][-1]}\n")
        file.write(f"Accuracy Score: {accuracy}\n")
        file.write(f"F1 Score: {f1}\n")
        file.write(f"Classification Report:\n {classification_report(y_test, y_pred_binary)}\n")
        
        # Save complete training history
        file.write("Training History:\n")
        for key, value in history.history.items():
            file.write(f"{key}: {value}\n")

            # Save confusion matrix
        file.write("\nConfusion Matrix:\n")
        conf_mat = confusion_matrix(y_test, y_pred_binary)
        file.write(np.array2string(conf_mat) + '\n')

        file.write("\n################################################################################################ \n\n")


def is_better_performance(current_accuracy, current_f1, saved_accuracy, saved_f1):
    # Check if current performance is better than saved performance
    if current_accuracy > saved_accuracy or current_f1 > saved_f1:
        return True
    else:
        return False
    
################ MAIN FUNCTION ################
    
def to_save_model(model, history, loss, accuracy, f1, model_name, y_test, y_pred_binary, building_function):
    # Check if model performance is better than the saved one
    
    save_summary_and_results(model, history, loss, accuracy, f1, model_name, y_test, y_pred_binary, building_function)
    
    # if not os.path.exists(f"MODELS_SUMMARIES/{model_name}/dscrpt.txt"):
    #     # If no saved model exists, save the current model's summary and results
    #     save_summary_and_results(model, history, loss, accuracy, f1, model_name, y_test, y_pred_binary, building_function)    
    # else:
    #     # If saved model exists, compare the performance and structure
    #     with open(f"MODELS_SUMMARIES/{model_name}/dscrpt.txt", "r") as file:
    #         lines = file.readlines()
    #         saved_accuracy = float(lines[9].split(":")[1].strip())
    #         saved_f1 = float(lines[10].split(":")[1].strip())
    #         saved_structure = "".join(lines[:9])  # Extract saved model structure

    #     current_structure = inspect.getsource(building_function)  # Get current model structure
        
    #     if is_better_performance(accuracy, f1, saved_accuracy, saved_f1) or current_structure != saved_structure:
    #         # If current performance is better or structure has changed, overwrite the saved description
    #         save_summary_and_results(model, history, loss, accuracy, f1, model_name, y_test, y_pred_binary, building_function)
