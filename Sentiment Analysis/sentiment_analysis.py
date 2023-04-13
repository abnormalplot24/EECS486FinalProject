import csv

def count_words(review, words):
    count = 0
    for word in words:
        count += review.lower().count(word)
    return count

positive_words = ['good', 'great', 'excellent', 'fun', 'enjoyable', 'recommended']
negative_words = ['bad', 'terrible', 'poor', 'boring', 'disappointing', 'not recommended']

def classify_review(review):
    positive_count = count_words(review, positive_words)
    negative_count = count_words(review, negative_words)
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

# Test the classifier
test_classifier(test_file)