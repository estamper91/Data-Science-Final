#General Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Decision Tree Libraries
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

#KNN Libraries
from sklearn.neighbors import KNeighborsClassifier

#Helper Functions
from helper_functions import metric_calc

#loading whole dataset
mushrooms = pd.read_csv('/Data-Science-Final/data/preprocessed/mushrooms_cleaned.csv')
mushrooms = mushrooms.drop(['bruise_bleed','has_ring'],axis=1)

print(mushrooms.head())
print(mushrooms.describe())

quant_features = ['cap_diameter','stem_height','stem_width']
cat_features = ['season','habitat','cap_shape','cap_color','gill_color','stem_color']
all_features = quant_features+cat_features

#Setting up dummy variables
mushrooms = pd.get_dummies(mushrooms,columns=cat_features)
print(mushrooms.columns)
#splitting the data into train/test/validate 
np.random.seed(470) #470 is the number of medicinal mushroom species

X = mushrooms.drop(['class'],axis=1)
y = mushrooms['class']

X_train, X_raw, y_train, y_raw = train_test_split(X, y, test_size=0.3)
print("X_train",X_train.shape)

#To use in a later file
X_raw = pd.DataFrame(X_raw)
y_raw = pd.DataFrame(y_raw)

X_raw.to_csv('/Data-Science-Final/data/preprocessed/X_raw.csv',index=False)
y_raw.to_csv('/Data-Science-Final/data/preprocessed/y_raw.csv',index=False)

#Decision Tree
#Train the model
depth_lim = 10

model = DecisionTreeClassifier(criterion='entropy',max_depth=depth_lim)
model.fit(X_train,y_train)

y_pred_train = model.predict(X_train)

# Evaluate the Model 
print("Decision Trees")
metric_calc(y_train,y_pred_train,"Training")


#Decision Tree with No Depth Limit
"""
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train,y_train)

y_pred_train = model.predict(X_train)

# Evaluate the Model 
metric_calc(y_train,y_pred_train,"Training")
"""


# Plot the DT
plt.figure(figsize=(16,8))

plot_tree(model,class_names=['edible','poisonous'], filled=True) #class names need to be in the correct order as their indexes/values "(0 and 1)", filled = colors

plt.title(f'Features: {(quant_features+cat_features)}, Max Depth: {depth_lim})')
plt.savefig('Decision Tree.jpg')
plt.show()



#KNN Classifier

#data is already loaded, as are the necessary libraries

#initial analysis
#mushroom_plot = sns.scatterplot(data = mushrooms, x = 'stem_height',y = 'cap_diameter',hue = 'class')

#plt.show()

best_recall = 0
best_k= 0
#Starting KNN Analysis
for k in range(1,11):
    knn_spec = KNeighborsClassifier(n_neighbors=k,weights='uniform')
    knn_fit = knn_spec.fit(X_train, y_train)
    knn_prediction = knn_fit.predict(X_train)

    train_recall = metrics.recall_score(y_train,knn_prediction)
    print(f"K: {k}, Training Recall Score: {train_recall}")

    if train_recall>best_recall:
        best_recall=train_recall
        best_k=k

#print(knn_prediction)

#Training the Model
knn_spec = KNeighborsClassifier(n_neighbors=3)
knn_fit = knn_spec.fit(X_train, y_train)
knn_prediction = knn_fit.predict(X_train)

#Evaluated the Model
print("K Nearest Neighbors")
metric_calc(y_train,knn_prediction,"Training")


#Gaussian Theorem 

#data is already loaded, as are the necessary libraries

#exploration
#sns.histplot(data = mushrooms, x='stem_height', hue='class')
#plt.show()
#sns.histplot(data = mushrooms, x='stem_width', hue='class')
#plt.show()
#sns.histplot(data = mushrooms, x='cap_diameter', hue='class')
#plt.show()

#Training the Model
modelGB = GaussianNB()
modelGB.fit(X_train, y_train)

y_predGB = modelGB.predict(X_train)

#Evaluating the Model
print("Gaussian Naive Bayes")
metric_calc(y_train,y_predGB,"Training")
