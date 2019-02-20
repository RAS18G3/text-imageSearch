from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg19 import preprocess_input
import pandas as pd
import pickle
import random
import numpy as np
from PIL import Image
import csv
import json
import codecs

from sample_triplets import sample_triplets

from pre_process import create_tokenizer, word_embedding_weights

'''
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
'''

IMAGE_EXTENSION = '../Flicker8k_Images/'
BI_DIRECTIONAL = True

# Initially the the Images are loaded as 255 x 255, but later they are cropped to 244 x 244
LOAD_IMG_HEIGHT = 224
LOAD_IMG_WIDTH = 224

IMG_HEIGHT = 224
IMG_WIDTH = 224

SAMPLES_PER_BATCH = 10

if BI_DIRECTIONAL:
    assert SAMPLES_PER_BATCH%2 == 0, 'sample batch must be devidable by two, when using bidirectional model'
#WORD_EMBEDDING_SIZE = 300
#WORD2VEC_PATH = 'Data/ru/ru.bin.syn0.npy'


class Data_generator:

    def __init__(self, batch_size, pre_trained=False):
            
        self.pre_trained = pre_trained
        
        # Load the data fram containing the training data
        # The file path to the training data file
        file_path = 'Data/df_flickr_train_17_01_19.tsv'
        # Fetch data frame from csv file, the data is allocated on the cpu ram and does not affect gpu memory
        df_query_data = pd.read_csv(file_path, sep="\t") 
        #df_query_data = df_query_data.sample(1000)
        
        # Batch sampler, sample batch returns a two list of triplet tuples for the batch
        self.batch_sampler = sample_triplets(df_query_data)
        self.tokenizer = create_tokenizer(df_query_data)
        self.embedding_matrix = word_embedding_weights(self.tokenizer)

        # +2 since zero padding is used in the embedding layer
        self.vocabulary_size = len(self.tokenizer.word_index) + 2

        self.training_samples_per_batch = SAMPLES_PER_BATCH
        self.batch_size = batch_size
        self.batch_nr = 0
        
        # Number of batches for each epoch
        self.batch_per_epoch = self.batch_sampler.df_query_data.shape[0] // batch_size
        # print('The number of batches per epoch is: {}'.format(self.batch_per_epoch))
        
        ## Trubbel shooting ##
        #self.batch_stat_dict = self.create_stat_dict()

    def load_image_batch(self, image_names):
        images = np.zeros((len(image_names), IMG_HEIGHT, IMG_WIDTH, 3))
        for i, image_name in enumerate(image_names):
            image = load_img(IMAGE_EXTENSION + image_name, target_size=(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH))
            #image = self.random_img_augmentation(image)

            images[i, :, :, :] = image

        # do the normalization required for vgg19
        images = preprocess_input(images)
        return images

    def create_data_from_image_triplets(self, triplets):

        anchor_img_names = list()
        positive_annotation_batch = list()
        negative_annotation_batch = list()
        for (img, ann, neg_ann) in triplets:
            anchor_img_names.append(img)

            # Tokenize the annotations
            positive_annotation = self.tokenizer.texts_to_sequences([ann])[0]
            negative_annotation = self.tokenizer.texts_to_sequences([neg_ann])[0]
            # Zero pad annotations
            positive_annotation = pad_sequences([positive_annotation], maxlen=25)
            negative_annotation = pad_sequences([negative_annotation], maxlen=25)

            positive_annotation_batch.append(positive_annotation)
            negative_annotation_batch.append(negative_annotation)

        # Turn the annotation batches in to np arrays
        positive_annotation_batch = np.array(positive_annotation_batch)
        negative_annotation_batch = np.array(negative_annotation_batch)
        # Squeeze away one of the redundant dimensions created when creating the np array
        positive_annotation_batch = np.squeeze(positive_annotation_batch)
        negative_annotation_batch = np.squeeze(negative_annotation_batch)

        anchor_img_batch = self.load_image_batch(anchor_img_names)

        return anchor_img_batch, positive_annotation_batch, negative_annotation_batch

    def create_data_from_annotation_triplets(self, triplets):

        anchor_annotation_batch = list()
        positive_img_names = list()
        negative_img_names = list()
        for (ann, img, neg_img) in triplets:
            positive_img_names.append(img)
            negative_img_names.append(neg_img)

            # Tokenize the annotations
            anchor_annotation = self.tokenizer.texts_to_sequences([ann])[0]
            # Zero pad annotations
            anchor_annotation = pad_sequences([anchor_annotation], maxlen=25)
            anchor_annotation_batch.append(anchor_annotation)

        # Turn the annotation batches in to np arrays
        anchor_annotation_batch = np.array(anchor_annotation_batch)
        # Squeeze away one of the redundant dimensions created when creating the np array
        anchor_annotation_batch = np.squeeze(anchor_annotation_batch)

        positive_img_batch = self.load_image_batch(positive_img_names)
        negative_img_batch = self.load_image_batch(negative_img_names)

        return anchor_annotation_batch, positive_img_batch, negative_img_batch

    def improved_batch_generator(self, model):
        # batches for the following inputs are created: anchor_img, positive_annotation, negative_annotation,
        # anchor_text, positive_img, negative_img
        while True:

            # Check if it is time for new epoch, in such case shuffle the data and zero the batch number
            if self.batch_nr == self.batch_per_epoch:
                self.batch_nr = 0
                '''
                # Store the batch dict to file
                with open('model/' + 'dict_stat' + '.pkl', 'wb') as f:
                    pickle.dump(self.batch_stat_dict, f, pickle.HIGHEST_PROTOCOL)
                '''
            # Sample triplets for the batch
            image_triplets, annotation_triplets = self.batch_sampler.sample_batch(self.batch_size)

            # create the actual batches from the batch triplets
            anchor_img_batch, positive_annotation_batch, negative_annotation_batch = self.create_data_from_image_triplets(image_triplets)
            
            anchor_annotation_batch, positive_img_batch, negative_img_batch = self.create_data_from_annotation_triplets(annotation_triplets)
            
            '''
            # FOR TESTING
            # Increase batch number
            self.batch_nr += 1

            yield {'anchor_img': anchor_img_batch, 'pos_text': positive_annotation_batch,
                   'neg_text': negative_annotation_batch,
                   'anchor_text': anchor_annotation_batch,
                   'pos_img': positive_img_batch,
                   'neg_img': negative_img_batch}, np.zeros(self.training_samples_per_batch)

            '''

            if BI_DIRECTIONAL:
                # Predict the loss for the different triplet pairs
                img_loss, text_loss = model.predict({'anchor_img': anchor_img_batch, 'pos_text': positive_annotation_batch,
                                                    'neg_text': negative_annotation_batch,'anchor_text': anchor_annotation_batch,
                                                    'pos_img': positive_img_batch, 'neg_img': negative_img_batch})
            else:
                img_loss = model.predict({'anchor_img': anchor_img_batch, 'pos_text': positive_annotation_batch,
                                          'neg_text': negative_annotation_batch, 'anchor_text': anchor_annotation_batch,
                                          'pos_img': positive_img_batch, 'neg_img': negative_img_batch})

            
            # Select top img and text losses, to put in to batch
            img_batch_permutation = np.argsort(img_loss)[-self.training_samples_per_batch:]
            
            
            if BI_DIRECTIONAL:
                text_batch_permutation = np.argsort(text_loss)[-self.training_samples_per_batch:]
                
                end = len(img_batch_permutation)//2
                img_batch_permutation = img_batch_permutation[:end]
                text_batch_permutation = text_batch_permutation[:end]
                
                # Select the firts part of the batch based on the top img loss examples
                anchor_img_batch_1 = anchor_img_batch[img_batch_permutation]
                positive_annotation_batch_1 = positive_annotation_batch[img_batch_permutation]
                negative_annotation_batch_1 = negative_annotation_batch[img_batch_permutation]
                
                anchor_annotation_batch_1 = anchor_annotation_batch[img_batch_permutation]
                positive_img_batch_1 = positive_img_batch[img_batch_permutation]
                negative_img_batch_1 = negative_img_batch[img_batch_permutation]
                
                # Select the second part of the batch based on the top text loss examples
                anchor_img_batch_2 = anchor_img_batch[text_batch_permutation]
                positive_annotation_batch_2 = positive_annotation_batch[text_batch_permutation]
                negative_annotation_batch_2 = negative_annotation_batch[text_batch_permutation]

                anchor_annotation_batch_2 = anchor_annotation_batch[text_batch_permutation]
                positive_img_batch_2 = positive_img_batch[text_batch_permutation]
                negative_img_batch_2 = negative_img_batch[text_batch_permutation]
                
                # Concatinate the twoparts to make one batch
                anchor_img_batch = np.concatenate((anchor_img_batch_1, anchor_img_batch_2))
                positive_annotation_batch = np.concatenate((positive_annotation_batch_1, positive_annotation_batch_2))
                negative_annotation_batch = np.concatenate((negative_annotation_batch_1, negative_annotation_batch_2))
                
                anchor_annotation_batch = np.concatenate((anchor_annotation_batch_1, anchor_annotation_batch_2))
                positive_img_batch = np.concatenate((positive_img_batch_1, positive_img_batch_2))
                negative_img_batch = np.concatenate((negative_img_batch_1, negative_img_batch_2))

            else:
                # Select the batch based on the top img loss examples
                anchor_img_batch = anchor_img_batch[img_batch_permutation]
                positive_annotation_batch = positive_annotation_batch[img_batch_permutation]
                negative_annotation_batch = negative_annotation_batch[img_batch_permutation]
                
                anchor_annotation_batch = anchor_annotation_batch[img_batch_permutation]
                positive_img_batch = positive_img_batch[img_batch_permutation]
                negative_img_batch = negative_img_batch[img_batch_permutation]
            
            ## Try to find which images are used as input to the network ##
            #triplet_img_batch = self.order_triplets(image_triplets, img_batch_permutation)
            #img_loss = np.array(img_loss)
            #self.image_stat(triplet_img_batch)
            
            # Increase batch number
            self.batch_nr += 1

            yield {'anchor_img': anchor_img_batch, 'pos_text': positive_annotation_batch,
                   'neg_text': negative_annotation_batch, 'anchor_text': anchor_annotation_batch,
                   'pos_img': positive_img_batch,'neg_img': negative_img_batch}, np.zeros(self.training_samples_per_batch)

            #'''
    def get_embedding_matrix(self):
        return self.embedding_matrix
######### Batch statistics ###########################

    def create_stat_dict(self):
        batch_stat = dict()
        for (img, ann) in self.anchor_pairs:
            batch_stat[img] = 0
        return batch_stat

    def image_stat(self, triplets):
        for (img, ann, ann_n) in triplets:
            self.batch_stat_dict[img] = self.batch_stat_dict[img] + 1
            #print('batch: {}, image: {}, pos_ann: {}, neg_ann: {}'.format(self.batch_nr, img, ann, ann_n)) 

    def order_triplets(self, triplets, order):
        ordered_triplets = list()
        for i in order:
            ordered_triplets.append(triplets[i])

        return ordered_triplets
    
    ####### Not usedmethods ###########
    def random_img_augmentation(self, image):

        # Randomly set if image should be flipped
        if random.randint(0, 1) == 1:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Randomly pick one of five different crops
        center_h = (LOAD_IMG_HEIGHT - IMG_HEIGHT) // 2
        center_w = (LOAD_IMG_WIDTH - IMG_WIDTH) // 2
        w = IMG_WIDTH
        h = IMG_HEIGHT
        area = [(0, 0, w, h), (0, LOAD_IMG_HEIGHT - h, w, LOAD_IMG_HEIGHT),
                (LOAD_IMG_WIDTH - w, 0, LOAD_IMG_HEIGHT, IMG_HEIGHT),
                (LOAD_IMG_WIDTH - w, LOAD_IMG_HEIGHT - h, LOAD_IMG_WIDTH, LOAD_IMG_HEIGHT),
                (center_w, center_h, center_w + w, center_h + h)
                ]

        rand_crop = random.randint(0, 4)
        image = image.crop(area[rand_crop])

        return image

if __name__ == '__main__':
    data_gen = Data_generator(20)
    batch_gen = data_gen.improved_batch_generator()
    batch = next(batch_gen)[0]
    print(batch)
    print(batch['anchor_brand'].shape)
    print(batch['anchor_brand'])
    print(batch['anchor_text'].shape)
    print(batch['anchor_text'])