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
[]
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
[14,14,20,14,22,22,27,35,35,35,35,35,35,35,35,45,45,45,45,35,45,45,45,45,45,45,45,45,14,20,35]
## Hyperparameters ##
### cell_ids ###
[35]
### lineNumbers ###
[74]
### source ###
``` 
from sklearn.naive_bayes import MultinomialNB

 ```
### values ###
"alpha,fit_prior"
### description ###
{"from sklearn.naive_bayes import multinomialnb":"undefined            'alpha': {\r\n                'type': 'number',\r\n                'distribution':'loguniform',\r\n                'minimumForOptimizer': 1e-10,\r\n                'maximumForOptimizer': 1.0,\r\n                'default': 1.0,\r\n                'description': 'Additive (Laplace/Lidstone) smoothing parameter'},\r\n            'fit_prior': {\r\n                'type': 'boolean',\r\n                'default': True,\r\n                'description': 'Whether to learn class prior probabilities or not.'},\r\n"}
## Miscellaneous ##
### cell_ids ###
[12,14,16,18]
### cells ###
"[object Object][object Object][object Object][object Object]"
### lineNumbers ###
[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
### source ###
``` 
#%matplotlib inline
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
[22,27,28,30,32,45]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128]
### source ###
``` 
import pylab as pl # useful for drawing graphics

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
["def count_data(labels, categories):    c = Counter(categories)\n    cont = dict(c)\n    tot = sum(list(cont.values()))\n    d = {[object Object]:[object Object],[object Object]:[object Object],[object Object]:[object Object]}\nprint(pd.DataFrame(d))\nprint(\"total \\t\",tot)\n    return cont","def categories_pie_plot(cont, tit):    global labels\n    sizes = [cont[l], continue]\npl.pie(sizes,explode=(0, 0, 0, 0),labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)\npl.title(tit)\npl.show()","def count_data(labels, categories):    c = Counter(categories)\n    cont = dict(c)\n    tot = sum(list(cont.values()))\n    d = {[object Object]:[object Object],[object Object]:[object Object],[object Object]:[object Object]}\nprint(pd.DataFrame(d))\nprint(\"total \\t\",tot)\n    return cont","def categories_pie_plot(cont, tit):    global labels\n    sizes = [cont[l], continue]\npl.pie(sizes,explode=(0, 0, 0, 0),labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)\npl.title(tit)\npl.show()","def categories_pie_plot(cont, tit):    global labels\n    sizes = [cont[l], continue]\npl.pie(sizes,explode=(0, 0, 0, 0),labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)\npl.title(tit)\npl.show()","def count_data(labels, categories):    c = Counter(categories)\n    cont = dict(c)\n    tot = sum(list(cont.values()))\n    d = {[object Object]:[object Object],[object Object]:[object Object],[object Object]:[object Object]}\nprint(pd.DataFrame(d))\nprint(\"total \\t\",tot)\n    return cont"]
### figures ###
### description ###
""
### outputs ###
Wall time: 1.06 s
,  category    news   percent
,0        b   81238  0.274738
,1        e  106844  0.361334
,2        m   31930  0.107984
,3        t   75681  0.255945
,total 	 295693

## Data Cleaning ##
### cell_ids ###
[20,43,49]
### cells ###
"[object Object][object Object][object Object]"
### lineNumbers ###
[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,99,100,101,130,131,132,133,134,135,136,137,138,139,140,141,142,143]
### source ###
``` 
from collections import Counter

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
### description ###
""
### outputs ###
  category    news   percent
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

## Evaluation ##
### cell_ids ###
[35,36,38,40,46,50]
### cells ###
"[object Object][object Object][object Object][object Object][object Object][object Object]"
### lineNumbers ###
[71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,129,144,145]
### source ###
``` 
from sklearn.feature_extraction.text import CountVectorizer 
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
### description ###
""
### outputs ###
Wall time: 27.1 s
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

