import csv
from collections import defaultdict
import random

negative_words = ['immersive', 'engaging', 'entertaining', 'addicting', 'satisfying',
                  'enjoyable', 'impressive', 'thrilling', 'captivating', 'exhilarating',
                  'innovative', 'polished', 'challenging', 'fun']


positive_words = ['buggy', 'glitchy', 'frustrating', 'unplayable', 'tedious', 'repetitive',
                  'uninspired', 'underwhelming', 'disappointing', 'broken', 'outdated',
                  'lackluster', 'shallow', 'bland', 'generic', 'mediocre']

sentiment_lexicon = {
    'immersive': -1.0,
    'engaging': -1.0,
    'entertaining': -1.0,
    'addicting': -1.0,
    'satisfying': -0.9,
    'enjoyable': -0.9,
    'impressive': -0.9,
    'thrilling': -0.9,
    'captivating': -0.9,
    'exhilarating': -0.8,
    'innovative': -0.8,
    'polished': -0.8,
    'challenging': -0.8,
    'fun': -0.8,
    'neutral': 0.0,
    'mediocre': 0.6,
    'substandard': 0.6,
    'glitchy': 0.6,
    'inferior': 0.6,
    'depressing': 0.7,
    'dreary': 0.7,
    'disappointing': 0.7,
    'lousy': 0.7,
    'shoddy': 0.7,
    'unsatisfactory': 0.7,
    'pitiable': 0.8,
    'pitiful': 0.8,
    'miserable': 0.8,
    'unacceptable': 0.8,
    'buggy': 0.8,
    'broken': 0.8,
    'wretched': 0.8,
    'pathetic': 0.8,
    'disgusting': 0.8,
    'vile': 0.8,
    'gruesome': 0.8,
    'ghastly': 0.9,
    'repulsive': 0.9,
    'atrocious': 0.9,
    'disastrous': 0.9,
    'dreadful': 0.9,
    'terrible': 0.9,
    'tragic': 0.9,
    'unplayable': 0.9,
    'tedious': 0.9,
    'repetitive': 0.9,
    'uninspired': 0.9,
    'underwhelming': 0.9,
    'outdated': 0.9,
    'lackluster': 0.9,
    'shallow': 0.9,
    'bland': 0.9,
    'generic': 0.9,
    'frustrating': 1.0,
    'disappointing': 1.0,
    'appalling': 1.0,
    'horrible': 1.0,
    'awful': 1.0
}

def count_words(review, words, weights):
    count = 0
    for word in words:
        count += weights[word] * review.lower().count(word)
    return count

def classify_review(review, weights):
    positive_count = count_words(review, positive_words, weights)
    negative_count = count_words(review, negative_words, weights)
    if positive_count == 0 and negative_count == 0:
        return 'Unknown'
    elif abs(positive_count) > abs(2 * negative_count):
        return 'Helpful'
    elif abs(negative_count) > abs(2 * positive_count):
        return 'Not Helpful'
    else:
        return 'Unknown'

def classifier(train_file, weights):
    with open(train_file, 'r') as f:
        reader = csv.DictReader(f)
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for row in reader:
            predicted = classify_review(row['review'], weights)
            actual = row['helpful']
            if predicted == 'Helpful' and int(actual) >= 1:
                true_positives += 1
            elif predicted == 'Helpful' and int(actual) == 0:
                false_positives += 1
            elif predicted == 'Not Helpful' and int(actual) >= 1:
                false_negatives += 1
            elif predicted == 'Not Helpful' and int(actual) == 0:
                true_negatives += 1
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall}

train_file = "Data/steam_reviews_training.csv"
test_file = "Data/steam_reviews_testing.csv"
# Assign weights to the words based on the sentiment lexicon
best_weights = None
best_accuracy = 0
for i in range(10):
    # Assign weights to the words based on the sentiment lexicon
    
    weights = defaultdict(float)
    for word in positive_words + negative_words:
        weights[word] = sentiment_lexicon[word] * random.uniform(0, 1000)

    # Train the classifier and print the results
    results = classifier(train_file, weights)
    print(f"Weighting Scheme {i + 1} - Results: {results}")

    # Keep track of the best weights and accuracy
    if results['accuracy'] > best_accuracy:
        best_weights = weights
        best_accuracy = results['accuracy']
        
results = classifier(test_file, best_weights)

print(f"Accuracy: {results['accuracy']}")
print(f"Precision: {results['precision']}")
print(f"Recall: {results['recall']}")
