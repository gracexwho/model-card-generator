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
[2]
### source ###
``` 
undefinedfrom sklearn.datasets import load_iris
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
[2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,9,11,14]
## Hyperparameters ##
### cell_ids ###
[11]
### lineNumbers ###
[66]
### source ###
``` 
from sklearn.linear_model import LogisticRegression
 ```
### values ###
[]
### description ###
{"from sklearn.linear_model import logisticregression":"undefined        'solver': {\r\n          'description': \"\"\"Algorithm for optimization problem.\r\n\r\n- For small datasets, 'liblinear' is a good choice, whereas 'sag' and\r\n  'saga' are faster for large ones.\r\n- For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'\r\n  handle multinomial loss; 'liblinear' is limited to one-versus-rest\r\n  schemes.\r\n- 'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty\r\n- 'liblinear' and 'saga' also handle L1 penalty\r\n- 'saga' also supports 'elasticnet' penalty\r\n- 'liblinear' does not support setting penalty='none'\r\nNote that 'sag' and 'saga' fast convergence is only guaranteed on\r\nfeatures with approximately the same scale. You can\r\npreprocess the data with a scaler from sklearn.preprocessing.\"\"\",\r\n          'enum': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],\r\n          'default': 'liblinear'},\r\n        'penalty': {\r\n          'description': \"\"\"Norm used in the penalization.  \r\nThe 'newton-cg', 'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is\r\nonly supported by the 'saga' solver. If 'none' (not supported by the\r\nliblinear solver), no regularization is applied.\"\"\",\r\n          'enum': ['l1', 'l2'],\r\n          'default': 'l2'},\r\n        'dual': {\r\n          'description': \"\"\"Dual or primal formulation. \r\nDual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.\"\"\",\r\n          'type': 'boolean',\r\n          'default': False},\r\n        'C': {\r\n          'description':\r\n            'Inverse regularization strength. Smaller values specify '\r\n            'stronger regularization.',\r\n          'type': 'number',\r\n          'distribution': 'loguniform',\r\n          'minimum': 0.0,\r\n          'exclusiveMinimum': True,\r\n          'default': 1.0,\r\n          'minimumForOptimizer': 0.03125,\r\n          'maximumForOptimizer': 32768},\r\n        'tol': {\r\n          'description': 'Tolerance for stopping criteria.',\r\n          'type': 'number',\r\n          'distribution': 'loguniform',\r\n          'minimum': 0.0,\r\n          'exclusiveMinimum': True,\r\n          'default': 0.0001,\r\n          'minimumForOptimizer': 1e-05,\r\n          'maximumForOptimizer': 0.1},\r\n        'fit_intercept': {\r\n          'description':\r\n            'Specifies whether a constant (bias or intercept) should be '\r\n            'added to the decision function.',\r\n          'type': 'boolean',\r\n          'default': True},\r\n        'class_weight': {\r\n          'anyOf': [\r\n            { 'description': 'By default, all classes have weight 1.',\r\n              'enum': [None]},\r\n            { 'description': \"\"\"Uses the values of y to automatically adjust weights inversely \r\nproportional to class frequencies in the input data as \"n_samples / (n_classes * np.bincount(y))\".\"\"\",\r\n              'enum': ['balanced']},\r\n            { 'description': 'Weights associated with classes in the form \"{class_label: weight}\".',\r\n              'type': 'object',\r\n              'additionalProperties': {'type': 'number'},\r\n              'forOptimizer': False}],\r\n        'multi_class': {\r\n          'description':\"\"\"Approach for handling a multi-class problem.\r\nIf the option chosen is 'ovr', then a binary problem is fit for each\r\nlabel. For 'multinomial' the loss minimised is the multinomial loss fit\r\nacross the entire probability distribution, *even when the data is\r\nbinary*. 'multinomial' is unavailable when solver='liblinear'.\r\n'auto' selects 'ovr' if the data is binary, or if solver='liblinear',\r\nand otherwise selects 'multinomial'.\"\"\",\r\n          'enum': ['ovr', 'multinomial', 'auto'],\r\n          'default': 'ovr'},\r\nset to 'liblinear' regardless of whether 'multi_class' is specified or\r\n              'solver': {'not': {'enum': ['newton-cg', 'sag', 'lbfgs']}}}},\r\n            'properties': {'penalty': {'enum': ['l2']}}}]},\r\n            'properties': {'dual': {'enum': [False]}}},\r\n              'penalty': {'enum': ['l2']},\r\n              'solver': {'enum': ['liblinear']}}}]},\r\n              'multi_class': {'not': {'enum': ['multinomial']}}}},\r\n              'solver': {'not': {'enum': ['liblinear']}}}}]}]}\r\n"}
## Miscellaneous ##
### cell_ids ###
[1,2]
### cells ###
"[object Object][object Object]"
### lineNumbers ###
[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
### source ###
``` 
#%matplotlib inline

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
[3,8]
### cells ###
"[object Object][object Object]"
### lineNumbers ###
[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,50,51,52,53,54,55,56,57,58]
### source ###
``` 
def plot_model_decision(model, proba=False):
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
["def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def treeviz(tree):    dot_data = StringIO()\nexport_graphviz(tree,out_file=dot_data,feature_names=['petal (cm)', 'sepal (cm)'],class_names=iris.target_names,filled=True,rounded=True,special_characters=True)\n    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())\n    return Image(graph.create_png())","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')"]
### figures ###
![id7.jpg](../example/iris.ipynb/id7.jpg)
### description ###
""
### outputs ###

## Data Cleaning ##
### cell_ids ###
[4,5,6,7]
### cells ###
"[object Object][object Object][object Object][object Object]"
### lineNumbers ###
[43,44,45,46,47,48,49]
### source ###
``` 
iris = load_iris()
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
[9,10,11,12,13,14,15,16,17,18]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78]
### source ###
``` 
from sklearn.tree import DecisionTreeClassifier

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
![id9.jpg](../example/iris.ipynb/id9.jpg)
![id10.jpg](../example/iris.ipynb/id10.jpg)
![id17.jpg](../example/iris.ipynb/id17.jpg)
### description ###
""
### outputs ###

## Evaluation ##
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

