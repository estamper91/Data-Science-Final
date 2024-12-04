#General Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#KNN Libraries
from sklearn.neighbors import KNeighborsClassifier

#Helper Functions
import sys
sys.path.append('/path/to/Data-Science-Final/utils')
from helper_functions import metric_calc

np.random.seed(470)

#importing necessary variables/functions
X_raw = pd.read_csv("/Data-Science-Final/data/preprocessed/X_raw.csv")
y_raw = pd.read_csv("/Data-Science-Final/data/preprocessed/y_raw.csv")
y_raw = np.ravel(y_raw)
print(X_raw.shape,y_raw.shape)


X_val, X_test, y_val, y_test = train_test_split(X_raw, y_raw, test_size=0.33)
print(X_val.shape,X_test.shape)


#Decision Tree Validation
depth_lim=10

model = DecisionTreeClassifier(criterion='entropy',max_depth=depth_lim)
model.fit(X_val,y_val)

y_pred_val = model.predict(X_val)


dt_prob = model.predict_proba(X_val)[:,1]
dt_fpr, dt_tpr, dt_thresholds = metrics.roc_curve(y_val, dt_prob)
dt_roc_auc = metrics.auc(dt_fpr, dt_tpr)
print(dt_roc_auc)

dt_CM = confusion_matrix(y_val,y_pred_val).ravel()
print(dt_CM)
"""
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_val,y_val)

y_pred_val = model.predict(X_val)

dt_prob = model.predict_proba(X_val)[:,1]
dt_fpr, dt_tpr, dt_thresholds = metrics.roc_curve(y_val, dt_prob)
dt_roc_auc = metrics.auc(dt_fpr, dt_tpr)
print(dt_roc_auc)
"""

# Evaluate the Model 
print("Decision Tree")
metric_calc(y_val,y_pred_val,"Validation")

#KNN Validation
best_k = 3  #use the one from the training set

model = KNeighborsClassifier(n_neighbors=best_k)
knn_fit = model.fit(X_val, y_val)
knn_prediction = knn_fit.predict(X_val)

knn_prob = model.predict_proba(X_val)[:,1]
knn_fpr, knn_tpr, knn_thresholds = metrics.roc_curve(y_val, knn_prob)
knn_roc_auc = metrics.auc(knn_fpr, knn_tpr)
print(knn_roc_auc)

knn_CM = confusion_matrix(y_val,knn_prediction).ravel()
print(knn_CM)

#Evaluate the Model
print("K Nearest Neighbors")
metric_calc(y_val,knn_prediction,"Validation")

#Naive Bayes Validation
modelGB = GaussianNB()
modelGB.fit(X_val, y_val)

y_predGB = modelGB.predict(X_val)

NB_prob = modelGB.predict_proba(X_val)[:,1]
NB_fpr, NB_tpr, NB_thresholds = metrics.roc_curve(y_val, NB_prob)
NB_roc_auc = metrics.auc(NB_fpr, NB_tpr)
print(NB_roc_auc)

gb_CM = confusion_matrix(y_val,y_predGB).ravel()
print(gb_CM)

# Evaluate the model
print("Gaussian Naive Bayes")
metric_calc(y_val,y_predGB,"Validation")

#Computing ROC Curves for Each Model
plt.figure()
plt.plot(dt_fpr, dt_tpr, color='gold', lw=2, label='DT ROC curve (area = %0.2f)' % dt_roc_auc)
plt.plot(knn_fpr, knn_tpr, color='red', lw=2, label='KNN ROC curve (area = %0.2f)' % knn_roc_auc)
plt.plot(NB_fpr, NB_tpr, color='orange', lw=2, label='NB ROC curve (area = %0.2f)' % NB_roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Comparison between Models')
plt.legend()
plt.savefig('ROC Curve.jpg')
plt.show()

#Creating Confusion Matrices
print(dt_CM,knn_CM,gb_CM)

dt_CM = dt_CM.reshape(2,2)
knn_CM = knn_CM.reshape(2,2)
gb_CM = gb_CM.reshape(2,2)

dt_CM_display = ConfusionMatrixDisplay(confusion_matrix=dt_CM,display_labels=['Edible','Poisonous'])
dt_CM_display.plot(cmap='YlOrBr')
plt.xlabel('Predicted Edible or Poisonous')
plt.ylabel('Actually Edible or Poisonous')
plt.title("Decision Tree Confusion Matrix")
plt.savefig("Decision Tree Confusion Matrix.jpg")
plt.show()

knn_CM_display = ConfusionMatrixDisplay(confusion_matrix=knn_CM,display_labels=['Edible','Poisonous'])
knn_CM_display.plot(cmap='YlOrBr')
plt.xlabel('Predicted Edible or Poisonous')
plt.ylabel('Actually Edible or Poisonous')
plt.title("KNN Confusion Matrix")
plt.savefig("KNN Confusion Matrix.jpg")
plt.show()

gb_CM_display = ConfusionMatrixDisplay(confusion_matrix=gb_CM,display_labels=['Edible','Poisonous'])
gb_CM_display.plot(cmap='YlOrBr')
plt.xlabel('Predicted Edible or Poisonous')
plt.ylabel('Actually Edible or Poisonous')
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.savefig("Gaussian Naive Bayes Confusion Matrix.jpg")
plt.show()

#Best Model = KNN
#KNN on the Test Data
best_k = 3  

model = KNeighborsClassifier(n_neighbors=best_k)
knn_fit = model.fit(X_test, y_test)
knn_test_prediction = knn_fit.predict(X_test)

#Evaluate the Model
print("K Nearest Neighbors Test")
metric_calc(y_test,knn_test_prediction,"Test")
