import pandas as pd
import numpy as np
import preprocess
import vectorspace
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
genesis_ic = wn.ic(genesis, False, 0.0)
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class KNN_Classifer():
    def __init__(self, k=1, distance_type = 'path'):
        self.k = k
        self.distance_type = distance_type

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        self.x_test = x_test
        y_predict = []

        for i in range(len(x_test)):
            max_sim = 0
            max_index = 0
            for j in range(len(self.x_train)):
                temp = self.similarity_score(x_test[i], self.x_train[j])
                if temp > max_sim:
                    max_sim = temp
                    max_index = j
            y_predict.append(self.y_train[max_index])
        return y_predict
    
    def similarity_score(self, s1, s2, distance_type = 'path'):
        return SequenceMatcher(None, s1, s2).ratio()

def main():
    steam_train = pd.read_csv("Data/steam_reviews_testing.csv")
    steam_test = pd.read_csv("Data/steam_reviews_testing.csv")

    steam_train["helpful_binary"] = steam_train["helpful"]
    steam_test["helpful_binary"] = steam_test["helpful"]

    arr = []
    for line in steam_train["review"]:
        line = str(line)
        text = preprocess.removeSGML(line)
        text = preprocess.tokenizeText(text)
        text = preprocess.removeStopwords(text)
        text = preprocess.stemWords(text)
        arr.append(text)

    X_train = arr
    Y_train = list(steam_train["helpful_binary"])

    arr = []
    for line in steam_test["review"]:
        line = str(line)
        text = preprocess.removeSGML(line)
        text = preprocess.tokenizeText(text)
        text = preprocess.removeStopwords(text)
        text = preprocess.stemWords(text)
        arr.append(text)
    
    X_test = arr
    Y_test = list(steam_test["helpful_binary"])

    classifier = KNN_Classifer(k=1, distance_type='path')
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
    print(conf_matrix)
    print(precision_score(Y_test, Y_pred))
    print(accuracy_score(Y_test, Y_pred))
    print(recall_score(Y_test, Y_pred))

if __name__ == "__main__":
    main()
