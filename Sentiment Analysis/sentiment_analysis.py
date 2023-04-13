import csv
from collections import defaultdict

sentiment_lexicon = {
        'excellent': 1.0,
        'great': 0.9,
        'good': 0.8,
        'fine': 0.6,
        'wonderful': 1.0,
        'amazing': 0.9,
        'awesome': 0.9,
        'fantastic': 0.9,
        'terrific': 0.8,
        'superb': 0.8,
        'phenomenal': 0.9,
        'splendid': 0.8,
        'marvelous': 0.8,
        'fabulous': 0.7,
        'neutral': 0.0,
        'bad': -0.8,
        'terrible': -0.9,
        'horrible': -1.0,
        'awful': -1.0,
        'abysmal': -0.9,
        'disgusting': -0.8,
        'dreadful': -0.9,
        'miserable': -0.8,
        'repulsive': -0.9,
        'ghastly': -0.9,
        'appalling': -1.0,
        'unpleasant': -0.6
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

# Define the positive and negative words
positive_words = ['excellent', 'great', 'good', 'fine', 'wonderful', 'amazing', 'awesome', 'fantastic', 'terrific', 'superb', 'phenomenal', 'splendid', 'marvelous', 'fabulous']
negative_words = ['bad', 'terrible', 'horrible', 'awful', 'abysmal', 'disgusting', 'dreadful', 'miserable', 'terrible', 'repulsive', 'ghastly', 'appalling', 'unpleasant']


# Assign weights to the words based on the sentiment lexicon
weights = defaultdict(float)
for word in positive_words + negative_words:
    weights[word] = sentiment_lexicon[word]

# Test the classifier
test_classifier(test_file)
