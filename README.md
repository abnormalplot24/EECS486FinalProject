# EECS486FinalProject README

The project data and code submission is organized in the following way.
In the home directory, we have:
1. Data: the data folder for the main data files of the project.
2. KNN Experiment 1: folder for scripts for the first KNN experiment.
3. LR_glove_experiment: folder for scripts and embedding data for the LR glove experiment.
4. Naive Bayes: folder for the scripts and stopword data for the naive bayes experiment.
5. Sentiment Analysis: folder for the sentiment-analysis-based experiment.
6. KNN Experiment 2: folder for scripts for the second KNN experiment.

## These are the specific instructions for executing each of the 5 experiments. <br>

### KNN Experiment 1

libraries needed to import:
- pandas
- nltk
- numpy
- sklearn

Also make sure to download:
- nltk.download('genesis')
- nltk.download('wordnet')

This experiment used a KNN model with K = 1 and Euclidean distance as the method to calculate distance.

To run the script:
1. Navigate to the KNN Experiment 1 folder
2. Run ```python3 model.py```<br>

### LR_glove_experiment

Overview <br>
logistic_regression_glove.py explores the combination of GloVE embeddings and logistic regression in order to predict whether a Steam review is helpful or not.

Requirements <br>
This may not be an exhaustive list, but to run the script on CAEN, the following additional libraries are needed:
1. pip3 install torchtext
2. pip3 install cleantext
3. pip3 install nltk
4. fully update scikit learn (1.2.2)

Other libraries used include matplotlib, pandas, numpy, and sklearn

Running the Script <br>
To run the script, navigate to the LR_glove_experiment folder and execute: <br>
```python3 logistic_regression_glove.py``` <br>
The expected outputs are accuracy and f-1 scores from train and test as well as a confusion matrix graph display. <br>

### Naive Bayes on a normalized steam review dataset

A common problem with datasets is dealing with skewed or unclean data. This combination of cleaning script and intuitive classification algorithm helps alliveate some of this issue,
and allows the user to achieve a recall rate of over 15% while keeping accuracy around 80%. This specific example helps classify steam reviews using this database:
https://www.kaggle.com/datasets/jummyegg/rawg-game-dataset
It performs a sentiment analysis on the dataset using Naive Bayes, and removing 90% of non-helpful reviews of the training set, in order to normalize the data. 
A cleaned dataset, containing only the helpfulness score and preprocessed english reviews can be obtained by running the cleaning algorithm.

**Installation**
Multiple libraries are required to run this script. Below are listed the libraries along with their respective versions:
numpy ver. (1.24.2)
scipy ver. (1.10.1)
csv
langdetect ver. (1.0.9)
random
sys

Program was originally run on Windows 10 using Visual Studio Code ver. 1.77

**Usage**
Cleaning will only work on the original dataset (see introduction). It can be run in Linux by going to the top directory and running python3 cleaning.py will result in a cleaned csv titled english_cleaned.csv. This dataset is needed to proporly run Naive_Bayes_Normalizer.py. For your convinience, a copy of english_cleaned.csv should be provided in the github repository.
The analyzer allows more customizability. It takes two arguments, the first being the training split you would like for the data, and the second being the number of non-helpful reviews you'd like to parse out of the dataset.
For example, to run the analyzer with an 80-20 training-test split, and to remove 20% of non-helpful reviews in order to normalize, go to the folder containing your program, dataset and stopwords text file and run:
python3 Naive_Bayes_Normalizer.py 0.8 0.2

The program will print the number of helpful reviews in training, the number of non-helpful reviews in training, the number of helpful reviews in test, the precision and recall, and the accuracy.

**Contributing**
This code is open to the public to be pulled or updated as desired at https://github.com/abnormalplot24/EECS486FinalProject.
All push and pull requests must be approved by repository owner, most likely me.

**License**
This software is free to use for both commercial and recreational use. Software can not be modified for malicious or illegal use. Software owner, me, has full rights to deny access to software without cause.

**Credits**
Credited to Devon Stein

I can be reached at devonsteincollege@gmail.com or on my github repo of abnormalplot24

**Additional Information**
I hope you have a nice day :) <br>

### Sentiment Analysis

Overview
The sentiment_analysis algorithm explores using a sentiment lexicon with unique weights to classify a review as 
helpful or not

Requirements
csv, collections, os, and random python libraries

Running the Script
1. Navigate to the Sentiment Analysis folder ```cd "Sentiment Analysis"```
2. ```python3 sentiment_analysis.py``` <br>

### KNN Experiment 2
Running the Script
1. Navigate to the KNN Experiment 2 folder
2. ```python3 knearest.py```

libraries needed to import:
sys
os
re
string
csv
from collections, OrderedDict Counter

This experiment used a KNN model with K = 5 and Euclidean distance as the method to calculate distance.
To run the script: python3 knearest.py

<br>
<br>

This is the end of the README
