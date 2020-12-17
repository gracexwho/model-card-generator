##  ##
### Filename ###
"Face_recognition.ipynb"
### cell_ids ###
[]
## Author ##
## Datasets ##
### description ###
""
### links ###
""
### cell_ids ###
[3]
### source ###
``` 
undefinedfrom sklearn.datasets import fetch_lfw_people
 ```
## References ##
### source ###
``` 

 ```
### cell_ids ###
[]
## Libraries Used ##
### lib ###
{"pandas":[],"numpy":["import numpy as np"],"matplotlib":["import matplotlib.pyplot as plt"],"sklearn":["from sklearn.datasets import fetch_lfw_people","from sklearn.model_selection import train_test_split","from sklearn.decomposition import PCA","from sklearn.model_selection import GridSearchCV","from sklearn.svm import SVC","from sklearn.metrics import classification_report","from sklearn.metrics import confusion_matrix"],"tensorflow":[],"pytorch":[],"OTHER":["from time import time","import logging"]}
### info ###
{"numpy":{"description":"Library numerical computation and N-dimensional arrays, mostly used in preprocessing.","link":"https://pandas.pydata.org/docs/"},"pandas":{"description":"Library for data analysis and manipulation, mostly used in preprocessing to create dataframes.","link":"https://numpy.org/doc/1.19/"},"matplotlib":{"description":"Library to create visualizations of data, mostly used for graphing.","link":"https://matplotlib.org/contents.html"},"sklearn":{"description":"Machine learning framework, built on NumPy, mostly used for model training and evaluation.","link":"https://scikit-learn.org/stable/user_guide.html"},"tensorflow":{"description":"Machine learning framework based on tensors, mostly used for model training and evaluation.","link":"https://www.tensorflow.org/api_docs"},"pytorch":{"description":"Machine learning frameork based on tensors, mostly used for model trainng and evaluation.","link":"https://pytorch.org/docs/stable/index.html"},"OTHER":{"description":""}}
### cell_ids ###
[2,2,2,2,2,3,3,2,2,2,2,2,2,2,7,2,10,2,2,2,2,2,15,15,2,2,2,19,20]
## Hyperparameters ##
### cell_ids ###
[10,15]
### lineNumbers ###
[54,77]
### source ###
``` 
from sklearn.decomposition import PCAfrom sklearn.svm import SVC
 ```
### values ###
[]
### description ###
{"from sklearn.svm import svc":"undefined            'kernel': {\r\n                'anyOf': [\r\n                {   'enum':['precomputed'], 'forOptimizer': False}, \r\n                {   'enum': ['linear', 'poly', 'rbf', 'sigmoid']},\r\n                {   'laleType': 'callable', 'forOptimizer': False}],\r\n                'default': 'rbf',\r\n                'description':\r\n                    'Specifies the kernel type to be used in the algorithm.'},\r\n            'degree': {\r\n                'type': 'integer',\r\n                'minimum': 0,\r\n                'minimumForOptimizer': 2,\r\n                'maximumForOptimizer': 5,\r\n                'default': 3,\r\n                'description': \"Degree of the polynomial kernel function ('poly').\"},\r\n            'gamma': {\r\n                'anyOf': [\r\n                {   'type': 'number',\r\n                    'minimum': 0.0,\r\n                    'exclusiveMinimum': True,\r\n                    'minimumForOptimizer': 3.0517578125e-05,\r\n                    'maximumForOptimizer': 8,\r\n                    'distribution': 'loguniform'},\r\n                {   'enum': ['auto', 'auto_deprecated', 'scale']}],\r\n                'default': 'auto_deprecated', #going to change to 'scale' from sklearn 0.22.\r\n                'description': \"Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.\"},\r\n            'shrinking': {\r\n                'type': 'boolean',\r\n                'default': True,\r\n                'description': 'Whether to use the shrinking heuristic.'},\r\n            'probability': {\r\n                'type': 'boolean',\r\n                'default': False,\r\n                'description': 'Whether to enable probability estimates.'},\r\n            'tol': {\r\n                'type': 'number',\r\n                'distribution': 'loguniform',\r\n                'minimum': 0.0,\r\n                'exclusiveMinimum': True,\r\n                'maximumForOptimizer': 0.01,\r\n                'default': 0.0001,\r\n                'description': 'Tolerance for stopping criteria.'},\r\n            'properties': {'kernel': {'enum': ['poly', 'sigmoid']}}},\r\n"}
## Miscellaneous ##
### cell_ids ###
[1]
### cells ###
"[object Object]"
### lineNumbers ###
[1,2]
### source ###
``` 
from __future__ import print_function
from time import time
 ```
### functions ###
[]
### figures ###
### description ###
""
### outputs ###

## Plotting ##
### cell_ids ###
[2,3,4,5]
### cells ###
"[object Object][object Object][object Object][object Object]"
### lineNumbers ###
[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
### source ###
``` 
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 10, 100)
y = np.sin(x) ** 2

plt.plot(y, 'o-')
plt.xlabel('x')
plt.ylabel('y = sin(x)')import logging
from sklearn.datasets import fetch_lfw_people

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data
n_samples, n_features = X.shape

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
print("height: %d pixels" % h)
print("width: %d pixels" % w)def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

        
plot_gallery(X, target_names[y], h, w)
 ```
### functions ###
["def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())"]
### figures ###
![id1.jpg](../example/Face_recognition.ipynb/id1.jpg)
![id4.jpg](../example/Face_recognition.ipynb/id4.jpg)
### description ###
""
### outputs ###
Total dataset size:
,n_samples: 1288
,n_features: 1850
,n_classes: 7
,height: 50 pixels
,width: 37 pixels

## Data Cleaning ##
### cell_ids ###
[7,10,11,12,13]
### cells ###
"[object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74]
### source ###
``` 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)from sklearn.decomposition import PCA

n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))X_train_reconstructed = np.dot(X_train_pca, pca.components_)
plot_gallery(X_train_reconstructed, target_names[y_train], h, w, n_row=1)plot_gallery(X_train, target_names[y_train], h, w, n_row=1)
 ```
### functions ###
[]
### figures ###
![id6.jpg](../example/Face_recognition.ipynb/id6.jpg)
![id8.jpg](../example/Face_recognition.ipynb/id8.jpg)
![id9.jpg](../example/Face_recognition.ipynb/id9.jpg)
### description ###
""
### outputs ###
Extracting the top 150 eigenfaces from 966 faces
,done in 0.379s
,Projecting the input data on the eigenfaces orthonormal basis
,done in 0.046s

## Preprocessing ##
### cell_ids ###
[]
### cells ###
[]
### lineNumbers ###
[]
### source ###
``` 

 ```
### functions ###
[]
### figures ###
### description ###
""
### outputs ###

## Model Training ##
### cell_ids ###
[15]
### cells ###
"[object Object]"
### lineNumbers ###
[75,76,77,78,79,80,81,82,83,84,85,86,87,88]
### source ###
``` 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1, 10, 100],
              'gamma': [0.0001, 0.001, 0.01, 0.1]}

clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, n_jobs=2)
clf = clf.fit(X_train_pca, y_train)

print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
 ```
### functions ###
[]
### figures ###
### description ###
""
### outputs ###
Fitting the classifier to the training set
,done in 4.402s
,Best estimator found by grid search:
,SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
,  decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
,  max_iter=-1, probability=False, random_state=None, shrinking=True,
,  tol=0.001, verbose=False)

## Evaluation ##
### cell_ids ###
[17,19,20]
### cells ###
"[object Object][object Object][object Object]"
### lineNumbers ###
[89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109]
### source ###
``` 
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=target_names))from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
 ```
### functions ###
[]
### figures ###
![id11.jpg](../example/Face_recognition.ipynb/id11.jpg)
### description ###
""
### outputs ###
Predicting people's names on the test set
,done in 0.059s
,                   precision    recall  f1-score   support
,
,     Ariel Sharon       0.64      0.73      0.68        22
,     Colin Powell       0.74      0.83      0.79        66
,  Donald Rumsfeld       0.62      0.84      0.71        25
,    George W Bush       0.96      0.79      0.87       123
,Gerhard Schroeder       0.74      0.83      0.78        24
,      Hugo Chavez       0.80      0.89      0.84        18
,       Tony Blair       0.80      0.75      0.78        44
,
,      avg / total       0.82      0.80      0.81       322
,
,[[16  3  1  0  0  1  1]
, [ 3 55  3  2  1  1  1]
, [ 2  0 21  0  0  1  1]
, [ 3 12  4 97  2  1  4]
, [ 0  1  2  0 20  0  1]
, [ 0  1  0  1  0 16  0]
, [ 1  2  3  1  4  0 33]]

