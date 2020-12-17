## # News Categorization using Multinomial Naive Bayes
 ##
### Filename ###
"News_Categorization_MNB.ipynb"
### cell_ids ###
0
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
https://www.linkedin.com/in/andres-soto-villaverde-36198a5/
https://www.kaggle.com/uciml/news-aggregator-dataset
http://archive.ics.uci.edu/ml
http://archive.ics.uci.edu/ml/datasets/News+Aggregator
http://pandas.pydata.org/
http://ipython.readthedocs.io/en/stable/interactive/magics.html#
https://ipython.org/ipython-doc/3/interactive/magics.html
https://docs.python.org/3/library/collections.html#counter-objects
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
http://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
http://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage
http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes
http://scikit-learn.org/stable/modules/pipeline.html
http://scikit-learn.org/stable/modules/classes.html
https://en.wikipedia.org/wiki/Precision_and_recall
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
### cell_ids ###
[0,1,3,4,9,10,16,18,20,25,33,33,33,33,36,38,41,47]
## Libraries Used ##
### lib ###
{"pandas":["import pandas as pd"],"numpy":["import numpy as np"],"matplotlib":["import matplotlib.pyplot as plt"],"sklearn":["from sklearn.utils import shuffle","from sklearn.feature_extraction.text import CountVectorizer","from sklearn.feature_extraction.text import TfidfTransformer","from sklearn.naive_bayes import MultinomialNB","from sklearn.pipeline import Pipeline","from sklearn import metrics"],"tensorflow":[],"pytorch":[],"OTHER":["from collections import Counter","import pylab as pl","import itertools"]}
### info ###
{"numpy":{"description":"Library numerical computation and N-dimensional arrays, mostly used in preprocessing.","link":"https://pandas.pydata.org/docs/"},"pandas":{"description":"Library for data analysis and manipulation, mostly used in preprocessing to create dataframes.","link":"https://numpy.org/doc/1.19/"},"matplotlib":{"description":"Library to create visualizations of data, mostly used for graphing.","link":"https://matplotlib.org/contents.html"},"sklearn":{"description":"Machine learning framework, built on NumPy, mostly used for model training and evaluation.","link":"https://scikit-learn.org/stable/user_guide.html"},"tensorflow":{"description":"Machine learning framework based on tensors, mostly used for model training and evaluation.","link":"https://www.tensorflow.org/api_docs"},"pytorch":{"description":"Machine learning frameork based on tensors, mostly used for model trainng and evaluation.","link":"https://pytorch.org/docs/stable/index.html"},"OTHER":{"description":""}}
### cell_ids ###
[2,2,2,2,2,3,3,2,2,2,2,2,2,2,7,2,10,2,2,2,2,2,15,15,2,2,2,19,20,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,10,10,12,2,2,3,2,5,5,7,15,12,12,13,15,15,15,15,17,19,19,19,15,19,19,19,19,19,19,19,19,2,3,15]
## Hyperparameters ##
### cell_ids ###
[10,15,10,13]
### lineNumbers ###
[54,77,66,74]
### source ###
``` 
from sklearn.decomposition import PCAfrom sklearn.svm import SVCfrom sklearn.linear_model import LogisticRegressionfrom sklearn.naive_bayes import MultinomialNB
 ```
### values ###
[]
### description ###
{"from sklearn.naive_bayes import multinomialnb":"undefined            'alpha': {\r\n                'type': 'number',\r\n                'distribution':'loguniform',\r\n                'minimumForOptimizer': 1e-10,\r\n                'maximumForOptimizer': 1.0,\r\n                'default': 1.0,\r\n                'description': 'Additive (Laplace/Lidstone) smoothing parameter'},\r\n            'fit_prior': {\r\n                'type': 'boolean',\r\n                'default': True,\r\n                'description': 'Whether to learn class prior probabilities or not.'},\r\n"}
## Miscellaneous ##
### cell_ids ###
[1,1,2,12,14,16,18]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[1,2,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206]
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
    return Image(graph.create_png())  #%matplotlib inline
import pandas as pd titles = [] # list of news titles
categories = [] # list of news categories
labels = [] # list of different categories (without repetitions)
nlabels = 4 # number of different categories
lnews = [] # list of dictionaries with two fields: one for the news and 
            # the other for its categorydef import_data():
    global titles, labels, categories
    # importing news aggregator data via Pandas (Python Data Analysis Library)
    news = pd.read_csv("uci-news-aggregator.csv")
    # function 'head' shows the first 5 items in a column (or
    # the first 5 rows in the DataFrame)
    print(news.head())
    categories = news['CATEGORY']
    titles = news['TITLE']
    labels = sorted(list(set(categories)))    #%time import_data()
 ```
### functions ###
[]
### figures ###
### description ###
""
### outputs ###
   ID                                              TITLE  \
,0   1  Fed official says weak data caused by weather,...   
,1   2  Fed's Charles Plosser sees high bar for change...   
,2   3  US open: Stocks fall after Fed official hints ...   
,3   4  Fed risks falling 'behind the curve', Charles ...   
,4   5  Fed's Plosser: Nasty Weather Has Curbed Job Gr...   
,
,                                                 URL          PUBLISHER  \
,0  http://www.latimes.com/business/money/la-fi-mo...  Los Angeles Times   
,1  http://www.livemint.com/Politics/H2EvwJSK2VE6O...           Livemint   
,2  http://www.ifamagazine.com/news/us-open-stocks...       IFA Magazine   
,3  http://www.ifamagazine.com/news/fed-risks-fall...       IFA Magazine   
,4  http://www.moneynews.com/Economy/federal-reser...          Moneynews   
,
,  CATEGORY                          STORY             HOSTNAME      TIMESTAMP  
,0        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM      www.latimes.com  1394470370698  
,1        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM     www.livemint.com  1394470371207  
,2        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM  www.ifamagazine.com  1394470371550  
,3        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM  www.ifamagazine.com  1394470371793  
,4        b  ddUyU0VZz0BRneMioxUPQVP6sIxvM    www.moneynews.com  1394470372027  
,Wall time: 7.72 s

## Plotting ##
### cell_ids ###
[2,3,4,5,3,8,22,27,28,30,32,45]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,159,160,161,162,163,164,165,166,167,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315]
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
plt.legend(loc='best');import pylab as pl # useful for drawing graphics

def categories_pie_plot(cont,tit):
    global labels
    sizes = [cont[l] for l in labels]
    pl.pie(sizes, explode=(0, 0, 0, 0), labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=90)
    pl.title(tit)
    pl.show()
    
categories_pie_plot(cont,"Plotting categories")from sklearn.utils import shuffle # Shuffle arrays in a consistent way

X_train = []
y_train = []
X_test = []
y_test = []

def split_data():
    global titles, categories
    global X_train, y_train, X_test, y_test,labels
    N = len(titles)
    Ntrain = int(N * 0.7)    
    # Let's shuffle the data
    titles, categories = shuffle(titles, categories, random_state=0)
    X_train = titles[:Ntrain]
    y_train = categories[:Ntrain]
    X_test = titles[Ntrain:]
    y_test = categories[Ntrain:]#%time split_data()cont2 = count_data(labels,y_train)categories_pie_plot(cont2,"Categories % in training set")import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:5.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    plt.show()
 ```
### functions ###
["def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_gallery(images, titles, h, w, n_row, n_col):\"Helper function to plot a gallery of portraits\"\nplt.figure(figsize=((1.8*n_col), (2.4*n_row)))\nplt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.9,hspace=0.35)\n    for i in range((n_row*n_col)):plt.subplot(n_row,n_col,(i+1))\nplt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)\nplt.title(titles[i],size=12)\nplt.xticks(())\nplt.yticks(())","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def treeviz(tree):    dot_data = StringIO()\nexport_graphviz(tree,out_file=dot_data,feature_names=['petal (cm)', 'sepal (cm)'],class_names=iris.target_names,filled=True,rounded=True,special_characters=True)\n    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())\n    return Image(graph.create_png())","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def plot_model_decision(model, proba):plt.figure(figsize=(8, 8))\n    xx, yy = np.meshgrid(np.linspace(0,9,100),np.linspace(0,9,100))\n    if proba:        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]    else:        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])\n    Z = Z.reshape(xx.shape)\n    cs = plt.contourf(xx,yy,Z,alpha=0.3)\n    for i, label in enumerate(iris.target_names):plt.scatter(X[(y==i)][:, 0],X[(y==i)][:, 1],label=label)\nplt.xlabel('Petal (cm)')\nplt.ylabel('Sepal (cm)')\nplt.xlim(0,9)\nplt.ylim(0,9)\nplt.legend(loc='best')","def count_data(labels, categories):    c = Counter(categories)\n    cont = dict(c)\n    tot = sum(list(cont.values()))\n    d = {[object Object]:[object Object],[object Object]:[object Object],[object Object]:[object Object]}\nprint(pd.DataFrame(d))\nprint(\"total \\t\",tot)\n    return cont","def categories_pie_plot(cont, tit):    global labels\n    sizes = [cont[l], continue]\npl.pie(sizes,explode=(0, 0, 0, 0),labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)\npl.title(tit)\npl.show()","def count_data(labels, categories):    c = Counter(categories)\n    cont = dict(c)\n    tot = sum(list(cont.values()))\n    d = {[object Object]:[object Object],[object Object]:[object Object],[object Object]:[object Object]}\nprint(pd.DataFrame(d))\nprint(\"total \\t\",tot)\n    return cont","def categories_pie_plot(cont, tit):    global labels\n    sizes = [cont[l], continue]\npl.pie(sizes,explode=(0, 0, 0, 0),labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)\npl.title(tit)\npl.show()","def categories_pie_plot(cont, tit):    global labels\n    sizes = [cont[l], continue]\npl.pie(sizes,explode=(0, 0, 0, 0),labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)\npl.title(tit)\npl.show()","def plot_confusion_matrix(cm, classes, title, cmap):\"\"\"\n    This function prints and plots the confusion matrix.\n    \"\"\"\nplt.imshow(cm,interpolation='nearest',cmap=cmap)\nplt.title(title)\nplt.colorbar()\n    tick_marks = np.arange(len(classes))\nplt.xticks(tick_marks,classes,rotation=45)\nplt.yticks(tick_marks,classes)\n    thresh = (cm.max()/2)\n    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):plt.text(j,i,'{:5.2f}'.format(cm[i, j]),horizontalalignment=\"center\",color=\"white\" if (cm[i, j]>thresh) else \"black\")\nplt.tight_layout()\nplt.ylabel('True label')\nplt.xlabel('Predicted label')\nplt.colorbar()\nplt.show()","def count_data(labels, categories):    c = Counter(categories)\n    cont = dict(c)\n    tot = sum(list(cont.values()))\n    d = {[object Object]:[object Object],[object Object]:[object Object],[object Object]:[object Object]}\nprint(pd.DataFrame(d))\nprint(\"total \\t\",tot)\n    return cont","def resume_data(labels, y_train, f1s):    c = Counter(y_train)\n    cont = dict(c)\n    tot = sum(list(cont.values()))\n    nlabels = len(labels)\n    d = {[object Object]:[object Object],[object Object]:[object Object],[object Object]:[object Object]}\nprint(pd.DataFrame(d))\nprint(\"total \\t\",tot)\n    return cont"]
### figures ###
![id1.jpg](../example/News_Categorization_MNB.ipynb/id1.jpg)
![id4.jpg](../example/News_Categorization_MNB.ipynb/id4.jpg)
![id21.jpg](../example/News_Categorization_MNB.ipynb/id21.jpg)
![id37.jpg](../example/News_Categorization_MNB.ipynb/id37.jpg)
![id41.jpg](../example/News_Categorization_MNB.ipynb/id41.jpg)
### description ###
""
### outputs ###
Total dataset size:
,n_samples: 1288
,n_features: 1850
,n_classes: 7
,height: 50 pixels
,width: 37 pixels
,Wall time: 1.06 s
,  category    news   percent
,0        b   81238  0.274738
,1        e  106844  0.361334
,2        m   31930  0.107984
,3        t   75681  0.255945
,total 	 295693

## Data Cleaning ##
### cell_ids ###
[7,10,11,12,13,4,5,6,7,20,43,49]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,152,153,154,155,156,157,158,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,286,287,288,317,318,319,320,321,322,323,324,325,326,327,328,329,330]
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
    X, y, test_size=50, random_state=42)from collections import Counter

def count_data(labels,categories):    
    c = Counter(categories)
    cont = dict(c)
    # total number of news
    tot = sum(list(cont.values()))     
    d = {
        "category" : labels,
        "news" : [cont[l] for l in labels],
        "percent" : [cont[l]/tot for l in labels]
    }
   
    print(pd.DataFrame(d))   
    print("total \t",tot) 
    
    return cont

cont = count_data(labels,categories)mat = metrics.confusion_matrix(y_test, predicted,labels=labels)
cm = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
cmdef resume_data(labels,y_train,f1s):
    c = Counter(y_train)
    cont = dict(c)
    tot = sum(list(cont.values()))
    nlabels = len(labels)
    d = {
        "category" : [labels[i] for i in range(nlabels)],
        "percent" : [cont[labels[i]]/tot for i in range(nlabels)],
        "f1-score" : [f1s[i] for i in range(nlabels)]
    }
   
    print(pd.DataFrame(d))   
    print("total \t",tot) 
    return cont
 ```
### functions ###
[]
### figures ###
![id6.jpg](../example/News_Categorization_MNB.ipynb/id6.jpg)
![id8.jpg](../example/News_Categorization_MNB.ipynb/id8.jpg)
![id9.jpg](../example/News_Categorization_MNB.ipynb/id9.jpg)
### description ###
""
### outputs ###
Extracting the top 150 eigenfaces from 966 faces
,done in 0.379s
,Projecting the input data on the eigenfaces orthonormal basis
,done in 0.046s
,  category    news   percent
,0        b  115967  0.274531
,1        e  152469  0.360943
,2        m   45639  0.108042
,3        t  108344  0.256485
,total 	 422419

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
![id23.jpg](../example/News_Categorization_MNB.ipynb/id23.jpg)
![id24.jpg](../example/News_Categorization_MNB.ipynb/id24.jpg)
![id31.jpg](../example/News_Categorization_MNB.ipynb/id31.jpg)
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
[17,19,20,35,36,38,40,46,50]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,316,331,332]
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

print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline 
from sklearn import metrics 
import numpy as np
import pprint

# lmats = [] # list of confussion matrix 
nrows = nlabels
ncols = nlabels
# conf_mat_sum = np.zeros((nrows, ncols))
# f1_acum = [] # list of f1-score

def train_test():
    global X_train, y_train, X_test, y_test, labels 
    #lmats, \
     #       conf_mat_sum, f1_acum, ncategories
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
    text_clf = text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    return predicted#%time predicted = train_test()metrics.accuracy_score(y_test, predicted)print(metrics.classification_report(y_test, predicted, target_names=labels))plot_confusion_matrix(cm, labels, title='Confusion matrix')f1s = metrics.f1_score(y_test, predicted, labels=labels, average=None)
cont3 = resume_data(labels,y_train,f1s)
 ```
### functions ###
[]
### figures ###
![id11.jpg](../example/News_Categorization_MNB.ipynb/id11.jpg)
![id48.jpg](../example/News_Categorization_MNB.ipynb/id48.jpg)
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
,Wall time: 27.1 s
,             precision    recall  f1-score   support
,
,          b       0.90      0.91      0.90     34729
,          e       0.95      0.97      0.96     45625
,          m       0.97      0.85      0.90     13709
,          t       0.90      0.90      0.90     32663
,
,avg / total       0.92      0.92      0.92    126726
,
,  category  f1-score   percent
,0        b  0.903839  0.274738
,1        e  0.959225  0.361334
,2        m  0.902814  0.107984
,3        t  0.903314  0.255945
,total 	 295693

