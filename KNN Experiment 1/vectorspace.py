# python3 vectorspace.py tfc tfx cranfieldDocs/ cranfield.queries
import os
import sys
import preprocess
from operator import itemgetter
import math

def indexDocument(document, doc_weight, query_weight, inverted_index, doc_index, doc_lengths):
    # preprocessing
    text = preprocess.removeSGML(document)
    text = preprocess.tokenizeText(text)
    text = preprocess.removeStopwords(text)
    text = preprocess.stemWords(text)

    # structure:
    # dictionary["term"] = [df, {doc_index: tf_doc_index}]
    for word in text:
        if word != '.' and word != ',':
            doc_lengths[doc_index] = len(text)
            if inverted_index.get(word) is not None:
                # calculating idf
                if inverted_index[word][1].get(doc_index) is not None:
                    inverted_index[word][1][doc_index] += 1
                else:
                    inverted_index[word][1][doc_index] = 1

                # increasing tf
                inverted_index[word][0] += 1
            else:
                inverted_index[word] = [1, {doc_index: 1}]

    return inverted_index, doc_lengths

def retrieveDocuments(query, inverted_index, doc_weight, query_weight, similarity, doc_lengths):
    # preprocessing
    text = preprocess.removeSGML(query)
    text = preprocess.tokenizeText(text)
    text = preprocess.removeStopwords(text)
    text = preprocess.stemWords(text)

    # remove query number
    query_id = text.pop(0)

    # remove punctuation
    for word in text:
        if word == '.':
            text.remove('.')
        if word == ',':
            text.remove(',')
    
    # finding document_ids that contain at least one of the query words
    doc_set = []
    for word in text:
        if inverted_index.get(word) is not None:
            doc_set.extend(list(inverted_index[word][1].keys()))

    # removing duplicates
    doc_set = [*set(doc_set)]
    similarity[query_id] = {}
    for doc in doc_set:
        similarity[query_id][doc] = 0

    for word in text:
        if inverted_index.get(word) is not None:
            for doc in list(inverted_index[word][1].keys()):
                norm = math.log(1400/inverted_index[word][0])

                if doc_weight == "tfc":
                    w_ij = inverted_index[word][1][doc] * norm * (1 / doc_lengths[doc])
                else:
                    # tfx
                    w_ij = inverted_index[word][1][doc] * norm
                
                if query_weight == "tfx":
                    w_iq = inverted_index[word][0] * norm
                else:
                    # tfc
                    w_iq = inverted_index[word][0] * norm * (1 / doc_lengths[doc])

                prod = w_ij * w_iq
                similarity[query_id][doc] += prod

    return similarity
    

def main():
    doc_weight = sys.argv[1]
    query_weight = sys.argv[2]
    cranfield_docs = os.path.dirname(os.path.realpath(__file__)) + "/" + sys.argv[3]
    cranfield_queries = os.path.dirname(os.path.realpath(__file__)) + "/" + sys.argv[4]

    inverted_index = {}
    doc_lengths = {}
    doc_index = 0

    # preprocessing and calculating tf + idf
    for filename in os.listdir(cranfield_docs):
        file = open(cranfield_docs + filename, 'r', encoding='ISO-8859-1')
        text = file.read()

        inverted_index, doc_lengths = indexDocument(text, doc_weight, query_weight, inverted_index, doc_index, doc_lengths)
        doc_index += 1

        file.close()

    # queries
    queries = open(cranfield_queries, 'r')
    lines = queries.readlines()
    similarity = {}
    for query in lines:
        similarity = retrieveDocuments(query, inverted_index, doc_weight, query_weight, similarity, doc_lengths)
        # break
    
    output_list = []
    # writing to cranfield.[doc_weight].[query_weight].output
    with open(f'cranfield.{doc_weight}.{query_weight}.output', 'w') as f:
        for query in list(similarity.keys()):
            for doc in list(similarity[query].keys()):
                output_list.append([query, doc, similarity[query][doc]])

            output_list = sorted(output_list, key=itemgetter(2), reverse=True)
            for item in output_list:
                print(f'{item[0]} {item[1]} {item[2]}', file=f)
            output_list = []

if __name__ == "__main__":
    main()