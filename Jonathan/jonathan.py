import csv
import math
import string
import itertools

# Node class to represent a node in the decision tree
class Node:
    def __init__(self, attr=None, value=None, result=None, branches=None):
        self.attr = attr    # attribute used for splitting
        self.value = value  # value of the attribute for this node
        self.result = result  # class label if leaf node, None otherwise
        self.branches = branches  # dictionary of branches (attribute value -> child node)

# Id3 algorithm
def id3(examples, target_attr, attrs):
    # Count the number of positive and negative examples in the target attribute
    num_pos = sum(1 for example in examples if example[target_attr] == "Recommended")
    num_neg = len(examples) - num_pos

    # If all examples are positive or negative, return a leaf node with the corresponding class label
    if num_pos == len(examples):
        return Node(result="Recommended")
    elif num_neg == len(examples):
        return Node(result="Not Recommended")

    # If there are no more attributes to split on, return a leaf node with the majority class label
    if not attrs:
        return Node(result="Recommended" if num_pos > num_neg else "Not Recommended")

    # Choose the attribute with the highest information gain
    best_attr = None
    best_gain = -1
    for attr in attrs:
        gain = information_gain(examples, target_attr, attr)
        if gain > best_gain:
            best_attr = attr
            best_gain = gain

    # Create a new node with the chosen attribute and its branches
    branches = {}
    for value in get_attribute_values(examples, best_attr):
        subset = [example for example in examples if example[best_attr] == value]
        if not subset:
            # If there are no examples with this attribute value, set the class label to the majority class
            branches[value] = Node(result="Recommended" if num_pos > num_neg else "Not Recommended")
        else:
            branches[value] = id3(subset, target_attr, [attr for attr in attrs if attr != best_attr])
    return Node(attr=best_attr, branches=branches)


# Information gain calculation
def information_gain(examples, target_attr, attr):
    # Calculate entropy of the target attribute
    entropy_target = entropy([example[target_attr] for example in examples])

    # Calculate entropy of the attribute by splitting on its values
    entropy_attr = 0
    for value in get_attribute_values(examples, attr):
        subset = [example for example in examples if example[attr] == value]
        if subset:
            entropy_attr += (len(subset) / len(examples)) * entropy([example[target_attr] for example in subset])

    # Calculate information gain as the difference between target entropy and attribute entropy
    return entropy_target - entropy_attr

# Entropy calculation
def entropy(labels):
    # Count the number of positive and negative labels
    num_pos = sum(1 for label in labels if label == "Recommended")
    num_neg = len(labels) - num_pos

    # Calculate entropy using the formula for binary classification
    if num_pos == 0 or num_neg == 0:
        return 0
    else:
        p_pos = num_pos / len(labels)
        p_neg = num_neg / len(labels)
        return -p_pos * math.log2(p_pos) - p_neg * math.log2(p_neg)

# Get the possible values of an attribute in a list of examples
def get_attribute_values(examples, attr):
    return set(example[attr] for example in examples)

# Preprocess the review text by removing punctuation and converting to lowercase
def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text

# Classify a review using the given decision tree
def classify_review(review, tree):
    # Traverse the decision tree until a leaf node is reached
    node = tree
    while node is None:
        value = preprocess_text(review.get(node.attr, ""))
        node = node.branches.get(value, node.branches.get(None))
    return node.result

def main():
    
    n = 1000
    # Read the first n rows from the training data CSV file
    with open("Data/steam_reviews_training.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        examples = list(itertools.islice(reader, n))

    # Train the decision tree
    target_attr = "recommendation"
    attrs = ["review"]
    tree = id3(examples, target_attr, attrs)

    # Read the first n rows from the testing data CSV file
    with open("Data/steam_reviews_testing.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        examples = list(itertools.islice(reader, n))

    # Test the decision tree and calculate accuracy score
    num_correct = 0
    for example in examples:
        predicted = classify_review(example, tree)
        actual = example[target_attr]
        if predicted == actual:
            num_correct += 1
    accuracy = num_correct / len(examples)
    print("Accuracy: {:.2%}".format(accuracy))
    
if __name__ == "__main__":
    main()