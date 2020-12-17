##  ##
### Filename ###
"iris.ipynb"
### cell_ids ###
[]
## Author ##
## Datasets ##
### description ###
""
### links ###
""
### cell_ids ###
[3,2]
### source ###
``` 
undefinedfrom sklearn.datasets import fetch_lfw_peoplefrom sklearn.datasets import load_iris
 ```
## References ##
### source ###
``` 

 ```
### cell_ids ###
[]
## Libraries Used ##
### lib ###
{"pandas":[],"numpy":["import numpy as np"],"matplotlib":["import matplotlib.pyplot as plt"],"sklearn":["from sklearn.model_selection import train_test_split","from sklearn.datasets import load_iris","from sklearn.tree import export_graphviz","from sklearn.tree import DecisionTreeClassifier","from sklearn.linear_model import LogisticRegression","from sklearn.linear_model import Perceptron"],"tensorflow":[],"pytorch":[],"OTHER":["from io import StringIO","import pydotplus","from IPython.display import Image"]}
### info ###
{"numpy":{"description":"Library numerical computation and N-dimensional arrays, mostly used in preprocessing.","link":"https://pandas.pydata.org/docs/"},"pandas":{"description":"Library for data analysis and manipulation, mostly used in preprocessing to create dataframes.","link":"https://numpy.org/doc/1.19/"},"matplotlib":{"description":"Library to create visualizations of data, mostly used for graphing.","link":"https://matplotlib.org/contents.html"},"sklearn":{"description":"Machine learning framework, built on NumPy, mostly used for model training and evaluation.","link":"https://scikit-learn.org/stable/user_guide.html"},"tensorflow":{"description":"Machine learning framework based on tensors, mostly used for model training and evaluation.","link":"https://www.tensorflow.org/api_docs"},"pytorch":{"description":"Machine learning frameork based on tensors, mostly used for model trainng and evaluation.","link":"https://pytorch.org/docs/stable/index.html"},"OTHER":{"description":""}}
### cell_ids ###
[2,2,2,2,2,3,3,2,2,2,2,2,2,2,7,2,10,2,2,2,2,2,15,15,2,2,2,19,20,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,10,10,12]
## Hyperparameters ##
### cell_ids ###
[10,15,10]
### lineNumbers ###
[54,77,66]
### source ###
``` 
from sklearn.decomposition import PCAfrom sklearn.svm import SVCfrom sklearn.linear_model import LogisticRegression
 ```
### values ###
"n_components,whiten,svd_solverkernel,degree,gamma,shrinking,probability,tolpenalty,dual,tol,C,fit_intercept,class_weight,solver,multi_class"
### description ###
{"from sklearn.linear_model import logisticregression":"undefined        'solver': {\r\n          'description': \"\"\"Algorithm for optimization problem.\r\n\r\n- For small datasets, 'liblinear' is a good choice, whereas 'sag' and\r\n  'saga' are faster for large ones.\r\n- For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'\r\n  handle multinomial loss; 'liblinear' is limited to one-versus-rest\r\n  schemes.\r\n- 'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty\r\n- 'liblinear' and 'saga' also handle L1 penalty\r\n- 'saga' also supports 'elasticnet' penalty\r\n- 'liblinear' does not support setting penalty='none'\r\nNote that 'sag' and 'saga' fast convergence is only guaranteed on\r\nfeatures with approximately the same scale. You can\r\npreprocess the data with a scaler from sklearn.preprocessing.\"\"\",\r\n          'enum': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],\r\n          'default': 'liblinear'},\r\n        'penalty': {\r\n          'description': \"\"\"Norm used in the penalization.  \r\nThe 'newton-cg', 'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is\r\nonly supported by the 'saga' solver. If 'none' (not supported by the\r\nliblinear solver), no regularization is applied.\"\"\",\r\n          'enum': ['l1', 'l2'],\r\n          'default': 'l2'},\r\n        'dual': {\r\n          'description': \"\"\"Dual or primal formulation. \r\nDual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.\"\"\",\r\n          'type': 'boolean',\r\n          'default': False},\r\n        'C': {\r\n          'description':\r\n            'Inverse regularization strength. Smaller values specify '\r\n            'stronger regularization.',\r\n          'type': 'number',\r\n          'distribution': 'loguniform',\r\n          'minimum': 0.0,\r\n          'exclusiveMinimum': True,\r\n          'default': 1.0,\r\n          'minimumForOptimizer': 0.03125,\r\n          'maximumForOptimizer': 32768},\r\n        'tol': {\r\n          'description': 'Tolerance for stopping criteria.',\r\n          'type': 'number',\r\n          'distribution': 'loguniform',\r\n          'minimum': 0.0,\r\n          'exclusiveMinimum': True,\r\n          'default': 0.0001,\r\n          'minimumForOptimizer': 1e-05,\r\n          'maximumForOptimizer': 0.1},\r\n        'fit_intercept': {\r\n          'description':\r\n            'Specifies whether a constant (bias or intercept) should be '\r\n            'added to the decision function.',\r\n          'type': 'boolean',\r\n          'default': True},\r\n        'class_weight': {\r\n          'anyOf': [\r\n            { 'description': 'By default, all classes have weight 1.',\r\n              'enum': [None]},\r\n            { 'description': \"\"\"Uses the values of y to automatically adjust weights inversely \r\nproportional to class frequencies in the input data as \"n_samples / (n_classes * np.bincount(y))\".\"\"\",\r\n              'enum': ['balanced']},\r\n            { 'description': 'Weights associated with classes in the form \"{class_label: weight}\".',\r\n              'type': 'object',\r\n              'additionalProperties': {'type': 'number'},\r\n              'forOptimizer': False}],\r\n        'multi_class': {\r\n          'description':\"\"\"Approach for handling a multi-class problem.\r\nIf the option chosen is 'ovr', then a binary problem is fit for each\r\nlabel. For 'multinomial' the loss minimised is the multinomial loss fit\r\nacross the entire probability distribution, *even when the data is\r\nbinary*. 'multinomial' is unavailable when solver='liblinear'.\r\n'auto' selects 'ovr' if the data is binary, or if solver='liblinear',\r\nand otherwise selects 'multinomial'.\"\"\",\r\n          'enum': ['ovr', 'multinomial', 'auto'],\r\n          'default': 'ovr'},\r\nset to 'liblinear' regardless of whether 'multi_class' is specified or\r\n              'solver': {'not': {'enum': ['newton-cg', 'sag', 'lbfgs']}}}},\r\n            'properties': {'penalty': {'enum': ['l2']}}}]},\r\n            'properties': {'dual': {'enum': [False]}}},\r\n              'penalty': {'enum': ['l2']},\r\n              'solver': {'enum': ['liblinear']}}}]},\r\n              'multi_class': {'not': {'enum': ['multinomial']}}}},\r\n              'solver': {'not': {'enum': ['liblinear']}}}}]}]}\r\n"}
## Miscellaneous ##
### cell_ids ###
[1,1,2]
### cells ###
"[object Object][object Object][object Object]"
### lineNumbers ###
[1,2,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130]
### source ###
``` 
from __future__ import print_function
from time import time#%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_irisfrom io import StringIO
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz


def treeviz(tree): 
    dot_data = StringIO()  
    export_graphviz(tree, out_file=dot_data,  
                    feature_names=['petal (cm)', 'sepal (cm)'],  
                    class_names=iris.target_names,  
                    filled=True, rounded=True,  
                    special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())  
 ```
### functions ###
[]
### figures ###
### description ###
""
### outputs ###

## Plotting ##
### cell_ids ###
[2,3,4,5,3,8]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,159,160,161,162,163,164,165,166,167]
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

        
plot_gallery(X, target_names[y], h, w)def plot_model_decision(model, proba=False):
    plt.figure(figsize=(8, 8))
    xx, yy = np.meshgrid(np.linspace(0, 9, 100),
                         np.linspace(0, 9, 100))

    if proba:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=0.3)


    for i, label in enumerate(iris.target_names):
        plt.scatter(X[y == i][:, 0], X[y == i][:, 1], label=label)

    plt.xlabel('Petal (cm)')
    plt.ylabel('Sepal (cm)')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.legend(loc='best');plt.figure(figsize=(8, 8))
for i, label in enumerate(iris.target_names):
    plt.scatter(X[y == i][:, 0], X[y == i][:, 1], label=label)

plt.xlabel('Petal (cm)')
plt.ylabel('Sepal (cm)')
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.legend(loc='best');
 ```
### functions ###
["def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def treeviz(tree):    dot_data = StringIO()\nexport_graphviz(tree,out_file=dot_data,feature_names=['petal (cm)', 'sepal (cm)'],class_names=iris.target_names,filled=True,rounded=True,special_characters=True)\n    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())\n    return Image(graph.create_png())","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')"]
### figures ###
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
[7,10,11,12,13,4,5,6,7]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,152,153,154,155,156,157,158]
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
plot_gallery(X_train_reconstructed, target_names[y_train], h, w, n_row=1)plot_gallery(X_train, target_names[y_train], h, w, n_row=1)iris = load_iris()
iris.target_namesiris.feature_namesX = iris.data[:, [2, 0]]
y = iris.targetX_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=50, random_state=42)
 ```
### functions ###
[]
### figures ###
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
[15,9,10,11,12,13,14,15,16,17,18]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[75,76,77,78,79,80,81,82,83,84,85,86,87,88,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187]
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
print(clf.best_estimator_)from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)
treeviz(tree)plot_model_decision(tree)from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression().fit(X_train[y_train != 0], y_train[y_train != 0])
plot_model_decision(lr_model, proba=True)lr_model.coef_lr_model.intercept_from sklearn.linear_model import Perceptron

linear_model = Perceptron(n_iter=50)
linear_model.fit(X_train[y_train != 0], y_train[y_train != 0])linear_model.score(X_test[y_test != 0], y_test[y_test != 0])linear_model.coef_linear_model.intercept_plot_model_decision(linear_model)
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

