Readme for logistic_regression_glove.py

Overview
logistic_regression_glove.py explores the combination of GloVE embeddings and logistic regression in order to predict whether a Steam review is helpful or not.

Requirements
This may not be an exhaustive list, but to run the script on CAEN, the following additional libraries are needed:
1. pip3 install torchtext
2. pip3 install cleantext
3. pip3 install nltk

Running the Script
To run the script, execute:
python3 logistic_regression_glove.py
The expected outputs are accuracy and f-1 scores from train and test as well as a confusion matrix graph display.