# Zhuoqi Zhang
# czzq
# EECS 486

import pandas as pd
import numpy as np
import torchtext
import matplotlib.pyplot as plt
from cleantext import clean
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# An option to switch between predicting helpfulness and recommendation. 
# The group has decided that the main focus is helpfuless.
TARGET = 'helpful'
# Data filename
FPATH = '../Data/steam_reviews.csv'

# Embedding filename and embedding length
EMBED_PATH = 'smaller_glove.txt'
EMBED_DIM = 200

def yield_tokens(data_iterable):
    '''Generator helper function for build_vocab_from_iterator.
    Reference: https://pytorch.org/text/stable/vocab.html#build-vocab-from-iterator
    Input: iterable object
    '''
    for text in data_iterable:
        text = text.lower()
        yield word_tokenize(text)

def embed_glove(glove_filename, vocab):
    '''Create embedding dictionary where vocabularies get mapped to their GloVE embeddings.
	Input: filename of the glove embedding txt, vocab object
	Output: numpy array with vocabularies replaced with their respective embeddings
    '''
    # Using the glove6B100d, dimension is 300
    transformed_vocab = np.zeros((len(vocab), EMBED_DIM))
    with open(glove_filename) as gfile:
        for line in gfile:
            embedding = line.strip().split()
            # First item in line is the word token, the rest of the line is the embedding
            if vocab[embedding[0]]:
                transformed_vocab[vocab[embedding[0]]] = embedding[1:]
                
    return transformed_vocab

def main():
	'''Main training and testing function'''

	df = pd.read_csv(FPATH)

	# Preprocessing
	df.review = df.review.apply(lambda r: clean(r, no_line_breaks=True, no_punct=True, no_emoji=True))
	#df.review = df.review.apply(lambda r: ' '.join([w for w in r.split() if w not in stopwords.words('english')]))

	# Ignore empty reviews
	df.dropna(subset=['review'], inplace=True)
	df = df[df.review != '']

	# Review labels are skewed: undersample class 0 to match the number of class 1
	positive = df[df.helpful > 0]
	negative_undersampled = df[df.helpful == 0].sample(35235, random_state=1)

	# Recommendation labels are fairly balanced, so create finalized dataframe based on our purpose
	if TARGET == 'recommendation':
	    balanced_df = df.copy()
	    balanced_df['label'] = (balanced_df['recommendation'] == 'Recommended').astype(int)
	else:
	    balanced_df = pd.concat([positive, negative_undersampled], axis=0)
	    balanced_df['label'] = (balanced_df['helpful'] > 0).astype(int)

	balanced_df['review_length'] = balanced_df.review.apply(lambda x: len(x.split()))

	X_train, X_test, y_train, y_test = train_test_split(balanced_df.drop('label', axis=1), balanced_df.label, test_size=0.33, random_state=1)

	# Prepare additional features to attach after word embedding
	X_train_non_text = X_train[['hour_played', 'review_length']].copy().reset_index(drop=True)
	X_test_non_text = X_test[['hour_played', 'review_length']].copy().reset_index(drop=True)

	# Build vocabulary based on the TRAINING set
	# words that appear fewer than twice are treated as <unk> ('unknown')
	# Standard practice referenced at: https://pytorch.org/text/stable/vocab.html#id1
	vocab = torchtext.vocab.build_vocab_from_iterator(
	    yield_tokens(X_train.review), 
	    specials=['<unk>'], 
	    min_freq=3
	)
	vocab.set_default_index(vocab["<unk>"])

	# Create lookup for embedded vocabularies
	transformed_vocab = embed_glove(EMBED_PATH, vocab)


	# Aggregate embeddings of word tokens into a dense vector for the sentence
	# A sentence with k words has a corresponding (k, 300) array, take average to get the (1, 300) vector for the sentence.

	X_train_embedded = X_train.apply(lambda row: np.mean(transformed_vocab[vocab.forward(word_tokenize(row['review'].lower()))], axis=0), axis=1)
	temp = pd.DataFrame(X_train_embedded)
	X_train_embedded = pd.DataFrame(temp[0].to_list(), columns=[str(i) for i in range(EMBED_DIM)])

	X_test_embedded = X_test.apply(lambda row: np.mean(transformed_vocab[vocab.forward(word_tokenize(row['review'].lower()))], axis=0), axis=1)
	temp = pd.DataFrame(X_test_embedded)
	X_test_embedded = pd.DataFrame(temp[0].to_list(), columns=[str(i) for i in range(EMBED_DIM)])

	# Include playtime and review length info
	X_train_embedded[['h', 'l']] = X_train_non_text
	X_test_embedded[['h', 'l']] = X_test_non_text

	# Fit
	clf = LogisticRegression(solver='newton-cholesky', C=100)
	clf.fit(X_train_embedded, y_train)

	# Training perf
	train_preds = clf.predict(X_train_embedded)
	print('Train acc:', accuracy_score(y_train, train_preds))
	print('Train f-1:', f1_score(y_train, train_preds))

	# Testing perf
	test_preds = clf.predict(X_test_embedded)
	print('Test acc:', accuracy_score(y_test, test_preds))
	print('Test f-1:', f1_score(y_test, test_preds))
	print('Test precision:', precision_score(y_test, test_preds))
	print('Test recall:', recall_score(y_test, test_preds))

	# Testing confusion matrix
	disp = ConfusionMatrixDisplay.from_estimator(clf, X_test_embedded, y_test, normalize='true')
	plt.show()

if __name__ == "__main__":
	main()
