import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import itertools

def plot_confusion_matrix(y_test, y_pred_binary):
    cm = confusion_matrix(y_test, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot()

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
def plot_roc_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(y_test, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Precision-Recall Curve")
    plt.legend(loc='lower left')
    plt.show()


def plot_f1_score_threshold(y_test, y_pred):
    y_pred_flat = np.concatenate(y_pred)
    f1_scores = []
    thresholds_list = []

    # Calculate F1 scores for each threshold
    for threshold in np.linspace(0, 1, num=100):
        y_pred_binary = (y_pred_flat > threshold).astype(int)
        f1 = f1_score(y_test, y_pred_binary)
        f1_scores.append(f1)
        thresholds_list.append(threshold)

    # Plot F1 scores with respect to thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_list, f1_scores, linestyle='-')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    plt.grid(True)
    plt.show()