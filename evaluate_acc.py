from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img
from keras.applications.vgg19 import preprocess_input
import numpy as np
import pandas as pd
from scipy import spatial
import json
import codecs

# To allow more recursions
import sys
sys.setrecursionlimit(10000)

IMAGE_EXTENSION = '../Flicker8k_Images/'

IMG_HEIGHT = 224
IMG_WIDTH = 224


class evaluate_acc:

    def __init__(self, tokenizer, image_embedding_model, text_embedding_model):

        self.tokenizer = tokenizer

        # Set up the valuation models
        self.image_embedding_model = image_embedding_model
        self.text_embedding_model = text_embedding_model
        
        # Load the validation dict
        file_path = 'Data/df_flickr_val_17_01_19.tsv'
        # Fetch data frame from csv file, the data is allocated on the cpu ram and does not affect gpu memory
        self.df_query_data = pd.read_csv(file_path, sep="\t") 
        #self.df_query_data = df_query_data.sample(100)

        # Load the validation data
        self.images, self.annotations = self.load_validation_data()

        # Initialize the query tree from where the nearest neighbours will be taken
        self.encoding_tree = self.image_encoding_tree()

    # Evaluate the accuracy over the validation-set
    def top_1_5_10_accuracy(self):

        top_1 = 0
        top_5 = 0
        top_10 = 0

        nr_samples = 0
        for i, q in enumerate(self.annotations):
            
            # Create the query and brand vectors, for feeding in to the embedding model
            query = q.lower()
            annotation_tokenized = self.tokenizer.texts_to_sequences([query])[0]
            padded_annotation_tokenized = pad_sequences([annotation_tokenized], maxlen=25)
            
            # Feed tokenized query to the text embedding model
            query_embedding = self.text_embedding_model.predict(padded_annotation_tokenized)
            
            predictions = self.encoding_tree.query(query_embedding[0], 10)
            #print(predictions[1])
            if i in predictions[1]: top_10 += 1
            if i in predictions[1][0:5]: top_5 += 1
            if i == predictions[1][0]: top_1 += 1

            nr_samples += 1

        top_10 /= nr_samples
        top_5 /= nr_samples
        top_1 /= nr_samples

        #print(top_1, top_5, top_10)
        return top_1, top_5, top_10


############ START: LOAD DATA ##########################################

    # Run the validation imags through the image embedding model to get the embedding for each image
    def encode_images(self):
        
        start = 0
        load_size = 20
        nr_images = ((len(self.images) // load_size) * load_size)
        predictions = np.zeros((nr_images, 512))
        for end in range(load_size, nr_images+1, load_size):
            image_list = self.images[start:end]
            images = self.load_images(image_list)
            prediction = self.image_embedding_model.predict(images)
            predictions[start:end,:] = prediction
            start = end
        return predictions
    
    # Put the images in to a tree structure, for quicker retreival
    def image_encoding_tree(self):
        image_encoding = self.encode_images()
        # Put the image encodings in a tree structure, for better query
        encoding_tree = spatial.KDTree(image_encoding)

        return encoding_tree

    # Load the pixel values for the image file names
    def load_images(self, image_list):
        images = np.zeros((len(image_list), IMG_HEIGHT, IMG_WIDTH, 3))
        for i, image_name in enumerate(image_list):
            image = load_img(IMAGE_EXTENSION + image_name, target_size=(IMG_HEIGHT, IMG_WIDTH))
            # do the normalization required for vgg19
            #image = preprocess_input(image)
            images[i, :, :, :] = image
        # Not sure if this is correct!!!    
        images = preprocess_input(images)  
        return images
    
    # Make a list out of the queries and images from the validation data frame
    def load_validation_data(self):
        annotations = list()
        images = list()
        
        for index, row in self.df_query_data.iterrows():
            # Create the annotation for image by concatinating, query, param1 and param2
            ann = row['annotations']
            ann = ann.lower()
            annotations.append(ann)
            
            # Create image name from external_image_id number
            img_name = row['image_name']
            images.append(img_name)
         
        return images, annotations
            
############ END: LOAD DATA ##########################################

if __name__ == '__main__':

    images, annotations = load_validation_data()

    print(images)
    print(annotations)



    #evaluator = evaluate_acc()
    #evaluator.top_1_5_10_accuracy()

