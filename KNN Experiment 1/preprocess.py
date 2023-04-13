import sys
import os
import re
from porterstemmer import PorterStemmer
import random
import math

def removeSGML(input):
    CLEANR = re.compile('<.*?>') 
    cleantext = re.sub(CLEANR, '', input)
    return cleantext

def tokenizeText(input):
    # TODO: finish this function
    # remove whitespace then split by ',' and '.' keeping these characters
    tokenized = re.split(r"\s+|(,)|(\.)", input)

    # tokenized = re.split(r"\s+", input)
    # tokenized2 = []
    # for item in tokenized:
    #     comma_number = re.compile(r"[[0-9]+\,[0-9]+]+")
    #     decimal_number = re.compile(r"[[0-9]+\.[0-9]+]+")
    #     if comma_number.search(item):
    #         # print("comma_number")
    #         # print(re.split(r"([\d+\,\d+]+)", item))
    #         tokenized2.append(re.split(r"([[0-9]+\,[0-9]+]+)", item))
    #     elif decimal_number.search(item):
    #         # print("decimal_number")
    #         # print(re.split(r"([\d+\.\d+]+)", item))
    #         tokenized2.append(re.split(r"[[0-9]+\.[0-9]+]+", item))
    #     else:
    #         print(item)
    #         print(re.split(r"(\,)|(\.)", item))
    #         tokenized2.append(re.split(r"(\,)|(\.)", item))

    # remove None values and '' values
    tokenized = [i for i in tokenized if i is not None and i != '']
    return tokenized

def removeStopwords(input):
    # getting stopword file and removing those words that match
    path = os.path.dirname(os.path.realpath(__file__)) + "/stopwords"
    stopwords = open(path).read().split()
    input = [i for i in input if i not in stopwords]
    return input

def stemWords(input):
    # using PorterStemmer implementation
    output_array = []
    p = PorterStemmer()
    for f in input:
        output_array.append(p.stem(f, 0, len(f)-1))
    return output_array

def main():
    path = os.path.dirname(os.path.realpath(__file__)) + "/" + sys.argv[1]

    total_words = []
    
    for filename in os.listdir(path):
        file = open(path + filename, 'r', encoding='ISO-8859-1')
        text = file.read()
        text = removeSGML(text)
        text = tokenizeText(text)
        text = removeStopwords(text)
        text = stemWords(text)
        total_words.extend(text)
        file.close()
    
    count = 0
    dict = {}

    for word in total_words:
        if word != '.' and word != ',':
            count += 1
            if dict.get(word) is not None:
                dict[word] += 1
            else:
                dict[word] = 1

    sorted_words = sorted(dict.items(), key=lambda x:x[1], reverse=True )

    # writing results to preprocess.output
    with open('preprocess.output', 'w') as f:
        print(f'Words {len(total_words)}', file=f)
        print(f'Vocabulary {len(sorted_words)}', file=f)
        print(f'Top 50 Words', file=f)
        for i in range(0,50):
            if len(sorted_words) >= i+1:
                print(f'{sorted_words[i][0]} {sorted_words[i][1]}', file=f)

    # calculations for preprocess.answers
    min_words = 0.25 * len(total_words)
    print(f'Minimum words: {min_words}')
    count = 0
    total = 0
    for tup in sorted_words:
        total += tup[1]
        count += 1
        if total >= min_words:
            break
    print(f'Min words needed for 25%: {count}')

    # calculating K and beta

    # picking random sample
    random.seed(12345)
    set1 = random.sample(total_words, 10000)
    set2 = random.sample(total_words, 20000)

    count = 0
    dict1 = {}
    for word in set1:
        if word != '.' and word != ',':
            count += 1
            if dict1.get(word) is not None:
                dict1[word] += 1
            else:
                dict1[word] = 1
    
    V1 = len(dict1)
    N1 = len(set1)

    count = 0
    dict2 = {}
    for word in set2:
        if word != '.' and word != ',':
            count += 1
            if dict2.get(word) is not None:
                dict2[word] += 1
            else:
                dict2[word] = 1

    V2 = len(dict2)
    N2 = len(set2)

    beta = math.log((float(V1)/V2), (float(N1)/N2))
    print(f"beta = {beta}")
    print(f"K = {V1 / N1**beta}")

if __name__ == "__main__":
    main()