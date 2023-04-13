import csv
from collections import defaultdict

positive_words = ['excellent', 'great', 'good', 'wonderful', 'amazing', 'awesome', 
                  'fantastic', 'terrific', 'superb', 'marvelous', 'lovely', 'delightful', 
                  'incredible', 'outstanding', 'exceptional', 'brilliant', 'inspiring']

negative_words = ['bad', 'terrible', 'horrible', 'awful', 'disgusting', 'dreadful', 'miserable', 
                  'appalling', 'mediocre', 'disappointing', 'disastrous', 'tragic', 'atrocious', 
                  'lousy', 'inferior', 'unacceptable', 'unsatisfactory', 'shoddy', 'wretched', 
                  'pathetic', 'substandard']

sentiment_lexicon = {
    'wonderful': 1.0,
    'amazing': 1.0,
    'excellent': 1.0,
    'great': 0.9,
    'fantastic': 0.9,
    'awesome': 0.9,
    'terrific': 0.8,
    'superb': 0.8,
    'marvelous': 0.8,
    'incredible': 0.8,
    'outstanding': 0.8,
    'exceptional': 0.8,
    'brilliant': 0.8,
    'inspiring': 0.8,
    'delightful': 0.8,
    'lovely': 0.7,
    'good': 0.7,
    'neutral': 0.0,
    'mediocre': -0.6,
    'substandard': -0.6,
    'inferior': -0.6,
    'depressing': -0.7,
    'dreary': -0.7,
    'disappointing': -0.7,
    'lousy': -0.7,
    'shoddy': -0.7,
    'unsatisfactory': -0.7,
    'pitiable': -0.8,
    'pitiful': -0.8,
    'miserable': -0.8,
    'unacceptable': -0.8,
    'bad': -0.8,
    'wretched': -0.8,
    'pathetic': -0.8,
    'disgusting': -0.8,
    'vile': -0.8,
    'gruesome': -0.8,
    'ghastly': -0.9,
    'repulsive': -0.9,
    'atrocious': -0.9,
    'disastrous': -0.9,
    'dreadful': -0.9,
    'terrible': -0.9,
    'tragic': -0.9,
    'appalling': -1.0,
    'horrible': -1.0,
    'awful': -1.0
}

def count_words(review, words, weights):
    count = 0
    for word in words:
        count += weights[word] * review.lower().count(word)
    return count

def classify_review(review):
    positive_count = count_words(review, positive_words, weights)
    negative_count = count_words(review, negative_words, weights)
    if positive_count == 0 and negative_count == 0:
        return 'Unknown'
    elif positive_count >= 2 * negative_count:
        return 'Recommended'
    elif negative_count >= 2 * positive_count:
        return 'Not Recommended'
    else:
        return 'Unknown'

def test_classifier(test_file):
    with open(test_file, 'r') as f:
        reader = csv.DictReader(f)
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for row in reader:
            predicted = classify_review(row['review'])
            actual = row['recommendation']
            if predicted == 'Recommended' and actual == 'Recommended':
                true_positives += 1
            elif predicted == 'Recommended' and actual == 'Not Recommended':
                false_positives += 1
            elif predicted == 'Not Recommended' and actual == 'Recommended':
                false_negatives += 1
            elif predicted == 'Not Recommended' and actual == 'Not Recommended':
                true_negatives += 1
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

test_file = "Data/steam_reviews_testing.csv"

# Assign weights to the words based on the sentiment lexicon
weights = defaultdict(float)
for word in positive_words + negative_words:
    weights[word] = sentiment_lexicon[word]

# Test the classifier
test_classifier(test_file)
