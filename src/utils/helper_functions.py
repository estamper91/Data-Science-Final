#General Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Decision Tree Libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def metric_calc(y,y_pred,label):
    train_accuracy = metrics.accuracy_score(y,y_pred)
    train_precision = metrics.precision_score(y,y_pred)
    train_recall = metrics.recall_score(y,y_pred)
    train_f1 = metrics.f1_score(y,y_pred)

    print(f"---{label} Performance---")
    print(f"{label} Accuracy:", train_accuracy)
    print(f"{label} Precision:", train_precision)
    print(f"{label} Recall:", train_recall)
    print(f"{label} F1 Score:", train_f1)
