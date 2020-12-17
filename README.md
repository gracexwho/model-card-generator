# model-card-generator
Typescript tool for generating Model Cards for machine learning models.

https://www.npmjs.com/package/@gracexwho/model-card-generator

## Requisites
1. Make sure you have Python & pip & NodeJS installed
2. ```pip install graphviz```
3. [Install GraphViz to your computer here](https://graphviz.org/download/)
4. ```npm i modified-python-program-analysis```

This repository builds on top of [python-program-analysis](https://github.com/andrewhead/python-program-analysis) and [jupyter-cmu-project](https://github.com/yjiang2cmu/Jupyter-Notebook-Project)

## Installation
1. Clone the repository to your local folders
2. run ```npm install``` to install node package dependencies

### Generate Cleaned Notebooks
1. Navigate to the ```source``` directory
2. Run ```node clean_notebook.js ../assets/```
This should generate cleaned notebooks (notebooks separated into machine learning stages) and all analysis files. All notebooks are currently stored in /assets/

### Generate Model Card
1. Naviage to the ```source``` directory
2. Run ```node main.js ../assets/${notebook name}```
