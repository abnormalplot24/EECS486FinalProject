# Emir Erben (erben)
import sys
import os
import re
import string
import csv
from collections import OrderedDict 
from collections import Counter



CLEANR = re.compile('<.*?>') 


print("Getting a runtime error, so this algorithm doesn't return an output, as mentioned in the report")

def removeSGML(text):
    cleaned = text.lower() # convert text to lowercase
    cleaned = re.sub('\[.*?\]', '', cleaned) # remove square brackets and text within them
    cleaned = re.sub('[%s]' % re.escape(string.punctuation), '', cleaned) # remove punctuation
    cleaned = re.sub('\w*\d\w*', '', cleaned) # remove words containing numbers
    cleaned = re.sub(' +', ' ', cleaned) # remove extra spaces
    return(cleaned)
def tokenizeText(text):
    tokenized = {}
    words = text.split()
    for word in words:
        #do tokenization steps with if

        # if re.match("^.*(,)$", word):
        #     # re.split(r'(\W+)', word)
        #     tokenized.append(word[:-1])
        #     tokenized.append(',')
        # elif re.match("^.*[.]$", word):
        #     # Finds properly but the data already has . with spacing, so it detects abbreviations. Ask what I should do
        #     tokenized.append(word[:-1])
        #     tokenized.append('.')
        # elif re.match("^.*(')$", word[:-1]):
        #     if word == 'he\'s':
        #         tokenized.append('he')
        #         tokenized.append('is')
        #     if word == 'she\'s':
        #         tokenized.append('she')
        #         tokenized.append('is')
        #     if word == 'it\'s':
        #         tokenized.append('it')
        #         tokenized.append('is')
        #     else: 
        #         tokenized.append(word[:-2])
        #         tokenized.append(word[-2:])
        # else:
        #     tokenized.append(word)
        
        if word not in tokenized:
            tokenized[word] = len(tokenized)
        # character_number = 1
    return(tokenized)

def read_text_file(file_path):
    
    
    with open(file_path, 'r') as f:
        # print(f.read())
        text = f.read()
        text = text.lower()
        text = removeSGML(text)
        text = tokenizeText(text)
        # text = dict.fromkeys(text)
        
        text = list(OrderedDict.fromkeys(text)) 
        for word in text:
            for character in word:
                if character in string.punctuation:
                    word = word.replace(character,"") 
            vocabulary.append(word)
            if file_path[0:4] == "fake":
                fake_vocabulary.append(word)
            if file_path[0:4] == "true":
                true_vocabulary.append(word)


        
        
        # print(vocabulary)
        # print(vocabulary)

# iterate through all file

def trainNaiveBayes(file_path):
    global vocabulary
    vocabulary = []
    global fake_vocabulary
    fake_vocabulary = []
    global true_vocabulary
    true_vocabulary = []

    # These are the ones without repetition
    final_fake_vocab = []
    final_true_vocab = []
    os.chdir(file_path)
    for file in os.listdir():
    # Check whether file is in text format or not
    # print(file_path)
        # print(file)
        # call read text file function

        read_text_file(file)
        # For fake
            # How many times does each word appear in fake documents + 1
            # /
            # vocab size of all documents (total no of unique words) + total number of words in fake documents
    # print(totalWords)
    # print(totalVocab)
    final_vocab = []
    for word in vocabulary:
        for character in word:
            if character in string.punctuation:
                word = word.replace(character,"") 
        final_vocab.append(word)
    final_vocab = list(OrderedDict.fromkeys(final_vocab))
    
    vocab_size = len(final_vocab)
    fake_counts = Counter(fake_vocabulary)
    true_counts = Counter(true_vocabulary)

    final_fake_vocab = list(OrderedDict.fromkeys(fake_vocabulary))
    final_true_vocab = list(OrderedDict.fromkeys(true_vocabulary))

    # print(fake_counts)

    return vocab_size, fake_counts, true_counts, final_fake_vocab, final_true_vocab
    # use the len of these dictionaries on the next function for calculations



    # for word in final_vocab:
    #     n_k = 
    # print(vocab_size)

def testNaiveBayes(path):
    vocab_size, fake_counts, true_counts, final_fake_vocab, final_true_vocab = trainNaiveBayes(path)
    correct_answer = 0
    total_answer = 0
    # print(final_true_vocab)
    # os.chdir(path)
    writefile = open('naivebayes.output.' + path[:-1], 'w')
    word_and_prob_true = {}
    word_and_prob_fake = {}
    for file in os.listdir():
    # Check whether file is in text format or not
    # print(file_path)
        print(file)
        with open(file, 'r') as f:
            total_answer = total_answer + 1
            text = f.read()
            text = text.lower()
            text = removeSGML(text)
            text = tokenizeText(text)
            # text = dict.fromkeys(text)
            
            text = list(OrderedDict.fromkeys(text)) 
            # print(text)
            probability_of_true = 0
            probability_of_fake= 0
            for word in text:
                for character in word:
                    if character in string.punctuation:
                        word = word.replace(character,"") 
                # print(word)
                p_fake = fake_counts[word]
                p_fake = p_fake / (vocab_size + len(fake_vocabulary))
                word_and_prob_fake[word] = p_fake
                # print(p_fake)
                probability_of_fake += p_fake

                p_true = true_counts[word]
                p_true = p_true / (vocab_size + len(true_vocabulary))
                word_and_prob_true[word] = p_true
                probability_of_true += p_true
                # print(p_true)
            if probability_of_true > probability_of_fake:
                writefile.write(file + ' ' + 'true' +  '\n')
                if file[0:4] == "true":
                    correct_answer += 1

            else:
                writefile.write(file + ' ' + 'fake' +  '\n')
                if file[0:4] == "fake":
                    correct_answer += 1
            print(probability_of_fake)
            print(probability_of_true)

    writefile.write('Accuracy: ' + str(correct_answer/total_answer) + '\n')
    word_and_prob_true = sorted(word_and_prob_true.items(), key=lambda x:x[1])
    print(word_and_prob_true)
    # print(sorted(word_and_prob_true))

def createVectors(data, vocab):
    feature_vectors = []
    for text in data:
        words = text.split()
        vector = [0] * len(vocab)
        for word in words:
            if word in vocab:
                vector[vocab[word]] += 1
        feature_vectors.append(vector)
    return feature_vectors

def euclidean_distance(x1, x2):
    sum_of_squares = 0.0
    for i in range(len(x1)):
        sum_of_squares += (x1[i] - x2[i]) ** 2
    return sum_of_squares ** 0.5

def knn_regression(X_train, y_train, X_test, k):
    y_pred = []
    for x in X_test:
        distances = [euclidean_distance(x, x_train) for x_train in X_train]
        nearest_neighbors = sorted(range(len(distances)), key=lambda i: distances[i])[:k]
        neighbors_y = [y_train[i] for i in nearest_neighbors]
        print(nearest_neighbors)
        print(k)
        y_pred.append(sum(neighbors_y) / k)
    return y_pred

with open('../Data/steam_reviews_test.csv', 'r') as csvtest:
    csvtest = csv.reader(csvtest)
    test_data = []
    test_y = []
    header = next(csvtest)
    review_index = header.index('review')
    helpful_index = header.index('helpful')
    for row in csvtest:
        test_data.append(removeSGML(row[review_index]))
        test_y.append(row[helpful_index])



with open('../Data/steam_reviews_training.csv', 'r') as csvfile:
    allreview = []
    # create a CSV reader object
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    review_index = header.index('review')
    helpful_index = header.index('helpful')
    train_data = []
    train_y = []
    global i
    i = 0

    for row in csvreader:
        train_data.append(removeSGML(row[review_index]))
        train_y.append(row[helpful_index])
        review = row[review_index]
        review = removeSGML(review)
        # print(review)
        allreview.append(review)
        i+=1

    # print(allreview)
    vocab = ''.join(allreview)
    vocab = vocab.lower()

    vocab = tokenizeText(vocab)
    x_train = createVectors(train_data, vocab)
    x_test = createVectors(test_data, vocab)

    k = 5
    y_pred = knn_regression(x_train, train_y, x_test, k)
    print(y_pred)
    # print(x_train)
    
