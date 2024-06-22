import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Function to plot ROC curve
def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(labels, probs))
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
  
# Function to calculate evaluation metrics
def calculate_metrics(labels, predictions, probs):
    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, probs)
    consusion_matrix = confusion_matrix(labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{consusion_matrix}")

    return accuracy, balanced_acc, f1, auc

# Function to plot ROC curve
def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(labels, probs))
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
  
# Function to calculate evaluation metrics
def calculate_metrics(labels, predictions, probs):
    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, probs)
    consusion_matrix = confusion_matrix(labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{consusion_matrix}")

    return accuracy, balanced_acc, f1, auc

''' metrics '''
def find_optimal_threshold(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters:
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations

    Returns:
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    th = list(roc_t['threshold'])
    print(f'the optimal threshold is {th}')
    return list(roc_t['threshold'])

# Function to calculate evaluation metrics
def calculate_metrics_test(labels, predictions, probs, th1 = 0.7): #use probs
    th = find_optimal_threshold(labels, probs)
    accuracy = accuracy_score(labels, np.array(probs) > th)
    balanced_acc = balanced_accuracy_score(labels, np.array(probs) > th)
    f1 = f1_score(labels, np.array(probs) > th)
    auc = roc_auc_score(labels, probs)
    consusion_matrix = confusion_matrix(labels, np.array(probs) > th)
    kappa = cohen_kappa_score(labels, np.array(probs) > th)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{consusion_matrix}")
    print(f"Cohen Kappa: {kappa:.4f}")
    print(f"---with th1 ={th1} :")
    print(f"Accuracy: {accuracy_score(labels, np.array(probs) > th1):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(labels, np.array(probs) > th1):.4f}")
    print(f"F1 Score: {f1_score(labels, np.array(probs) > th1):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(labels, np.array(probs) > th1)}")
    print(f"Cohen Kappa: {cohen_kappa_score(labels, np.array(probs) > th1):.4f}")
    #evaluate(labels, predictions, show = True)
    return accuracy, balanced_acc, f1, auc
