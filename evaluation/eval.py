import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, cohen_kappa_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from plot_metric.functions import BinaryClassification
from matplotlib import pyplot as plt
  
''' images and plots'''
def generate_confusion_matrix_image(y_test_predicted, y_test, threshold, show, save_path = 'confusion_matrix.png'):
  np.set_printoptions(precision=2)
  titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", "true"),]
  for title, normalize in titles_options:
      disp = ConfusionMatrixDisplay.from_predictions(
          y_test.flatten(), y_test_predicted.flatten() >threshold, cmap=plt.cm.Blues, normalize=normalize)
      disp.ax_.set_title(title)
      print(title)
      print(disp.confusion_matrix)
      plt.savefig(save_path, bbox_inches='tight')
      if show:
        plt.show()

def plot_nice_roc_curve(y_test, y_test_predicted, show, save_path = 'roc_curve.png'):
  # Visualisation with plot_metric
  bc = BinaryClassification(y_test.flatten(), y_test_predicted.flatten(), labels=["Class 1", "Class 2"])
  plt.figure(figsize=(5,5))
  bc.plot_roc_curve()
  plt.savefig(save_path, bbox_inches='tight')
  if show:
    plt.show()


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
    return list(roc_t['threshold']) 
