import pandas as pd
import numpy as np
import math
import nltk
import re

# Load the data from the CSV files
train_df = pd.read_csv('Data/steam_reviews_training.csv')
test_df = pd.read_csv('Data/steam_reviews_testing.csv')

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're","you've", 
             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he','him', 'his', 'himself', 
             'she', "she's", 'her', 'hers', 'herself', 'it', "it's",'its', 'itself', 'they', 'them', 'their', 
             'theirs', 'themselves', 'what', 'which','who', 'whom', 'this', 'that', "that'll", 'these', 
             'those', 'am', 'is', 'are','was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
             'do','does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because','as', 'until', 
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against','between', 'into', 'through', 'during', 
             'before', 'after', 'above', 'below','to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
             'under', 'again','further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how','all', 
             'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such','no', 'nor', 'not', 'only', 
             'own', 'same', 'so', 'than', 'too', 'very', 's','t', 'can', 'will', 'just', 'don', "don't", 'should', 
             "should've", 'now', 'd','ll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
             'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",'haven', "haven't", 'isn', 
             "isn't", 'ma', 'mightn', "mightn't", 'mustn',"mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', 
             "shouldn't",'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# define a function to perform text preprocessing
def preprocess_text(text):
    # convert text to lowercase
    text = text.lower()
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # remove custom stopwords
    text = " ".join(word for word in text.split() if word not in stopwords)
    return text

# apply the preprocessing function to the "review" column of both dataframes
train_df['review'] = train_df['review'].apply(preprocess_text)
test_df['review'] = test_df['review'].apply(preprocess_text)

# Convert 'Recommended' to 1 and 'Not Recommended' to 0
train_df['label'] = train_df['recommendation'].map({'Recommended': 1, 'Not Recommended': 0})
test_df['label'] = test_df['recommendation'].map({'Recommended': 1, 'Not Recommended': 0})

def entropy(column):
    elements, counts = np.unique(column, return_counts=True)
    entropy = -np.sum([(counts[i] / np.sum(counts)) * math.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])

    elements, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[feature] == elements[i]).dropna()[target]) for i in range(len(elements))])

    info_gain = total_entropy - weighted_entropy
    return info_gain

def id3(data, original_data, features, target):
    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]

    elif len(data) == 0:
        return np.unique(original_data[target])[np.argmax(np.unique(original_data[target], return_counts=True)[1])]

    elif len(features) == 0:
        return data[target].value_counts().idxmax()

    else:
        best_feature = max(features, key=lambda feature: information_gain(data, feature, target))
        tree = {best_feature: {}}

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            sub_tree = id3(sub_data, data, [feat for feat in features if feat != best_feature], target)
            tree[best_feature][value] = sub_tree

        return tree

def predict(tree, example):
    for feature in tree.keys():
        value = example[feature].values[0]
        if value not in tree[feature].keys():
            return 0
        if isinstance(tree[feature][value], dict):
            return predict(tree[feature][value], example)
        else:
            return tree[feature][value]

features = train_df.columns.drop(['review', 'title', 'recommendation', 'label'])
target = 'label'

# Train the classifier
decision_tree = id3(train_df, train_df, features, target)

# Test the classifier
test_df['prediction'] = test_df.apply(lambda row: predict(decision_tree, row.to_frame().transpose()), axis=1)

# Calculate and print the accuracy score
accuracy = sum(test_df['label'] == test_df['prediction']) / len(test_df)
print("Accuracy: ", accuracy)
