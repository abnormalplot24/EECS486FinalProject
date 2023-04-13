import csv
import numpy as np
import scipy
import langdetect
import sys
import random

global stop_words #stop words taken from language identification project.
global category_counts
global word_counts
global accuracy_array
global precision_array
global recall_array

# Count the frequency of each word in each category
def count_words(review, category):
    category_counts[category] += 1
    for word in review:
        if word not in word_counts[category]:
            word_counts[category][word] = 0
        word_counts[category][word] += 1
    return word_counts, category_counts

# Calculate the probabilities of each word in each category
def calculate_word_probabilities( alpha=1):
    word_probabilities = {category: {} for category in category_counts}
    for category in category_counts:
        total_words = sum(word_counts[category].values())
        for word in word_counts[category]:
            word_probabilities[category][word] = (word_counts[category][word] + alpha) / (total_words + alpha * len(word_counts[category]))
    return word_probabilities

# Calculate the prior probabilities of each category
def calculate_prior_probabilities():
    total_categories = sum(category_counts.values())
    return {category: category_counts[category] / total_categories for category in category_counts}

# Predict the category of a text
def predict(text, word_probabilities, prior_probabilities, categories):
    category_scores = {category: prior_probabilities[category] for category in categories}
    for category in categories:
        for word in text:
            if word in word_probabilities[category]:
                category_scores[category] *= word_probabilities[category][word]

    return category_scores
if __name__ == "__main__":
    precision_array = []
    recall_array = []
    accuracy_array = []
    train_split = float(sys.argv[1])
    test_split = 1 - train_split
    normalizer = 0
    while normalizer < 1:
        data_text_test = []
        data_text_train = []


        final_dataset = []

        #x = np.array(range(16296))

        #    plt.title("Line graph")
        #    plt.xlabel("X axis")
        #    plt.ylabel("Y axis")


        #    final_score = sorted(np.log(final_score))
        #   plt.plot(x, (final_score))
        #    plt.show()
        #    plt.savefig('logged.png')

        #    header = [ 'funny', 'helpful', 'hour_played', 'recommendation', 'review', 'title']
        #   writer.writerow(header)
        decile = 0

        categories = ['helpful', 'not_helpful']
        word_counts = {category: {} for category in categories}
        category_counts = {category: 0 for category in categories}
        column = []
        with open('english_cleaned.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            index = 0
            dummy = 0
            score = 0
            for row in reader:
                try:
                    score += float(row[0])
                except:
                    dummy = 1
                    #print(row[0])
                if index == 0:
                    index =1
                else:
                    column.append(float(row[0]))
                    if random.random() > train_split:
                        data_text_test.append(row)
                    else:
                        if float(row[0]) > 0 or random.random() > normalizer:
                            data_text_train.append(row)
            #print(score, "score")  
            #print(len(data_text_test), "test")
            # print(len(data_text_train), "train")

            helpful_cutoff = np.percentile(column, np.arange(0,100,10))
            #print(helpful_array)
            # print(helpful_cutoff)
            decile = 0
            #for decile in helpful_cutoff:
                #print(column, decile)
                
            index = 0
            helpful_count_2 = 0
            not_helpful_count_2 = 0
            for row in data_text_train:
                if index == 0:
                    index =1
                else:
                    if float(row[0]) > decile:
                        category = "helpful"
                        helpful_count_2 += 1
                    else:

                        category = "not_helpful"
                        not_helpful_count_2 += 1

                    count_words(row[1], category)
            print(helpful_count_2, " helpful in training ")
            print(not_helpful_count_2, "not helpful in training ")

            word_prob = calculate_word_probabilities()
            prior_prob = calculate_prior_probabilities()

            helpful_count = 0
            not_helpful_count = 0
            true_positive = 0 # we correctly guess a helpful item that was helpful
            false_positives = 0 # we guessed helpful, but it was not helpful
            false_negatives = 0 # we guess it was not helpful, but it actually was
            true_negative = 0
            index = 0
            real_helpful_in_test = 0
            answer = "helpful"
            for row in data_text_test:
                if index == 0:
                    index =1
                else:
                    prediction = predict(row[1], word_prob, prior_prob, categories)
                    #print(row[1], max(prediction))
                    #print(prediction["helpful"])
            
                    if(prediction["helpful"] > prediction["not_helpful"]):
                        helpful_count += 1
                        answer = "helpful"
                    else:
                        not_helpful_count += 1
                        answer = "not_helpful"
                    if answer == "helpful":
                        if float(row[0]) > 0:
                            true_positive += 1
                            real_helpful_in_test +=1
                        else:
                            false_positives +=1
                    else:
                        if float(row[0]) > 0:
                            false_negatives +=1 
                            real_helpful_in_test +=1
                        else:
                            true_negative +=1


            precision = true_positive / (true_positive + false_positives)
            recall = true_positive / (true_positive + false_negatives)
            print("We had ", real_helpful_in_test, "actual helpful reviews in test set" )
            print(precision, recall, "precision and recall")
            print("accuracy", (true_positive + true_negative)/ len(data_text_test)  )
            #print("train test split is", train_split, test_split)
            print("normalize and skip ", normalizer, "of non_helpful training reviews\n")
            accuracy_array.append((true_positive + true_negative)/ len(data_text_test))
            precision_array.append(precision)
            recall_array.append(recall)
            normalizer += 0.05
    print(precision_array, accuracy_array, recall_array)