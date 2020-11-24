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
[]
## References ##
### cell_ids ###
[]
## Hyperparameters ##
### cell_ids ###
[]
### cells ###
[]
### lineNumbers ###
[]
### values ###
""
## Miscellaneous ##
### cell_ids ###
[1,2]
### cells ###
"[object Object][object Object]"
### lineNumbers ###
[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
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
### functions ###
[]
### figures ###
### description ###
""
### outputs ###

