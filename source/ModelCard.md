# MODEL CARD #
### Model_Name ###
News_Categorization_MNB
## Author Info ##
Andrés Soto
## Dataset ##
uci-news-aggregator.csv
## References ##
https://www.linkedin.com/in/andres-soto-villaverde-36198a5/
http://archive.ics.uci.edu/ml
http://archive.ics.uci.edu/ml/datasets/News+Aggregator
http://pandas.pydata.org/
http://ipython.readthedocs.io/en/stable/interactive/magics.html#
https://ipython.org/ipython-doc/3/interactive/magics.html
https://docs.python.org/3/library/collections.html#counter-objects
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
http://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html

## Libraries Used ##
#### From the library [pandas](https://pandas.pydata.org/docs/) ####
import pandas as pd    START:3	 END:146

#### From the library [numpy](https://numpy.org/doc/1.19/) ####
import numpy as np    START:77	 END:146

#### From the library [matplotlib](https://matplotlib.org/contents.html) ####
import matplotlib.pyplot as plt    START:104	 END:129

#### From the library [sklearn](https://scikit-learn.org/stable/user_guide.html) ####
from sklearn.utils import shuffle    START:51	 END:64

from sklearn.feature_extraction.text import TfidfTransformer    START:73	 END:90

from sklearn.naive_bayes import MultinomialNB    START:74	 END:90

from sklearn.pipeline import Pipeline    START:75	 END:90

from sklearn import metrics    START:76	 END:146

#### From the library OTHER ####
import pylab as pl    START:40	 END:146

import itertools    START:103	 END:146

## Data Cleaning ##
### cell_ids ###
1
### lineNumbers ###
3,4,5,6,7,8,9
### functions ###

### figures ###
      
### description ###
```Fill out yourself```

## Preprocessing ##
### cell_ids ###
4
[object Object]
### lineNumbers ###
21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
### functions ###

### description ###
```Fill out yourself```

## Hyperparameters ##
### cell_ids ###

### lineNumbers ###

## Model Training ##
### cell_ids ###
10
### lineNumbers ###
73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98

### description ###
```Fill out yourself```

## Evaluation ##
### cell_ids ###
12
### lineNumbers ###
100,101
### description ###
```Fill out yourself```


### Figures ###
![id5](./image/id5.jpg)
![id9](./image/id9.jpg)
![id16](./image/id16.jpg)
