import pandas as pd
import numpy as np
import math
import preprocess

# Load the data from the CSV files
train_df = pd.read_csv('Data/steam_reviews_training.csv')
test_df = pd.read_csv('Data/steam_reviews_testing.csv')

# Apply text preprocessing
train_df['review'] = train_df['review'].apply(preprocess.preprocess_text)
test_df['review'] = test_df['review'].apply(preprocess.preprocess_text)

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
