import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
import operator
import pickle

WORD2VEC_npy_PATH = 'Data/ru/ru.bin.syn0.npy'
WORD2VEC_tsv_PATH = 'Data/ru/ru.tsv'
GLOVE_PATH = 'Data/glove/glove.6B.300d.txt'
WORD_EMBEDDING_SIZE = 300

PRE_TRAINED = True

def string_to_int(s):
    try:
        i = int(s)
        return True, i
    except:
        return False, None

def create_tokenizer(df_query_data):

    if PRE_TRAINED:
        # loading
        with open('model/tokenizer/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        # Creating dictionary with the number of occurances for each word
        word_dict = dict()
        for index, row in df_query_data.iterrows():
            query = row['annotations']
            query = query.lower()
            query_list = query.split(' ')
            for word in query_list:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
                    
        # Turning the dict key-value pairs in to tuple items, putting and ordering the items in a list
        sorted_word_dict = sorted(word_dict.items(), key=operator.itemgetter(1))
        sorted_word_dict.reverse()
        
        # List with the words ordered based on occurance
        sorted_word_list = [x[0] for x in sorted_word_dict]
        
        # Create a tokenizer instance
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(sorted_word_list)
        
        # +2 since zero padding is used in the embedding layer
        #vocabulary_size = len(tokenizer.word_index) + 2
        
    return tokenizer

def word_embedding_weights(tokenizer):
    
    missing_words = set()
    
    # load the entire embedding from file into a dictionary
    embeddings_index = dict()
    f = open(GLOVE_PATH, encoding='utf-8')
    for line in f:
        # splits on spaces
        values = line.split()
        # the word for the vector is the first word on the row
        word = values[0]
        # Extra the vector corresponding to the word
        vector = np.asarray(values[1:], dtype='float32')
        # Add word (key) and vector (value) to dictionary
        if word not in embeddings_index:
            embeddings_index[word] = vector
    f.close()
    
    vocabulary_size = len(tokenizer.word_index) + 2
    # Initialize an embedding matrix with shape vocab_size x word_vector_size
    embedding_matrix = np.zeros((vocabulary_size, WORD_EMBEDDING_SIZE))
    # Go through the tokenizer and for each index add the corresponding word vector to the row
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
        else:
            missing_words.add(word)

    return embedding_matrix

def save_embedding_matrix(embedding_matrix):
    np.save('./model/Embedding_weights/embedding_matrix.npy', embedding_matrix)

def load_embedding_matrix():
    return np.load('./model/Embedding_weights/embedding_matrix.npy')