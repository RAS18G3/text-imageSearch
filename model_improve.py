from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda, Embedding, BatchNormalization, LSTM, Flatten, Dropout, concatenate
from keras.applications.vgg19 import VGG19
from keras import backend
from keras.optimizers import Adam
import tensorflow as tf
import sys

from keras import backend as K
K.set_image_dim_ordering('tf')  

WORD_EMBEDDING_SIZE = 300

IMG_HEIGHT = 224
IMG_WIDTH = 224

BI_DIRECTIONAL = True

def tf_print(op, tensors, message=None):
    def print_message(x):
        sys.stdout.write(message + " %s\n" % x)
        return x

    prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
    with tf.control_dependencies(prints):
        op = tf.identity(op)
    return op

class triplet_loss_embedding_graph:

    def __init__(self, vocab_size, embedding_weights):

        # Could be passed right to the text embedding models, save memory?
        self.vocab_size = vocab_size
        self.word_embedding_weights = embedding_weights

        # The embedding models for images and text
        self.img_embedding_model = self.create_image_embedding_model()
        self.text_embedding_model = self.create_text_embedding_model()

        # Combined model of the embedding models, optimized based on bidirectional triplet loss
        self.model, self.batch_prediction_model = self.create_embedding_model()
        # Compile combined model
        self.model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.0002))
        print('Model summary')
        self.model.summary()
        self.batch_prediction_model._make_predict_function()
        #print('Batch prediction model summary')
        #self.batch_prediction_model.summary()

    # Putting the main model together
    def create_embedding_model(self):
        # Initialize the text and image encoding models
        img_embedding_model = self.img_embedding_model
        text_embedding_model = self.text_embedding_model
        text_embedding_model.summary()

        # Define input sizes
        image_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
        text_shape = (25,)

        # Define all inputs, Image anchor and positive/ negative
        anchor_image_example = Input(shape=image_shape, name='anchor_img')
        positive_image_example = Input(shape=image_shape, name='pos_img')
        negative_image_example = Input(shape=image_shape, name='neg_img')

        # Text anchor and positive/ negative
        anchor_text_example = Input(shape=text_shape, name='anchor_text')
        positive_text_example = Input(shape=text_shape, name='pos_text')
        negative_text_example = Input(shape=text_shape, name='neg_text')

        # Define the embedding models for all the different input types
        anchor_image_embedding = img_embedding_model(anchor_image_example)
        positive_text_embedding = text_embedding_model(positive_text_example)
        negative_text_embedding = text_embedding_model(negative_text_example)

        anchor_text_embedding = text_embedding_model(anchor_text_example)
        positive_image_embedding = img_embedding_model(positive_image_example)
        negative_image_embedding = img_embedding_model(negative_image_example)

        # Image loss layer
        img_loss_layer = Lambda(self.img_loss, output_shape=(None,), name='img_loss')([anchor_image_embedding,
                                                                                       positive_text_embedding,
                                                                                       negative_text_embedding])

        # Text loss layer
        text_loss_layer = Lambda(self.text_loss, output_shape=(None,), name='text_loss')([anchor_text_embedding,
                                                                                          positive_image_embedding,
                                                                                          negative_image_embedding])

        # Triplet loss layer
        triplet_layer = Lambda(self.bi_directional_triplet_loss, output_shape=(1,), name='triplet_loss')\
            ([img_loss_layer, text_loss_layer])

        if BI_DIRECTIONAL:
            print('Bi directional triplet loss graph')
            # Model used to calculate the loss over the image anchor and text query anchor
            batch_prediction_model = Model(inputs=[anchor_image_example, positive_text_example,
                                                   negative_text_example, anchor_text_example,
                                                   positive_image_example, negative_image_example],
                                           outputs=[img_loss_layer, text_loss_layer])
        else:
            print('Single directional triplet loss graph')
            # For the single direction loss function, calculating loss over image anchor
            batch_prediction_model = Model(inputs=[anchor_image_example, positive_text_example,
                                                   negative_text_example, anchor_text_example,
                                                   positive_image_example, negative_image_example],
                                           outputs=[img_loss_layer])

        # Create the final model
        triplet_model = Model(inputs=[anchor_image_example, positive_text_example, negative_text_example,
                              anchor_text_example, positive_image_example, negative_image_example],
                              outputs=[triplet_layer])

        return triplet_model, batch_prediction_model

    # Calculate the loss based on the image anchor, compared with positive and negative text query
    def img_loss(self, input):

        anchor_image, positive_text, negative_text = input

        # Printing input tensors for debug
        #anchor_image = tf_print(anchor_image, [anchor_image], 'anchor_image: ')
        #positive_text = tf_print(positive_text, [positive_text], 'positive_text: ')
        #negative_text = tf_print(negative_text, [negative_text], 'negative_text: ')

        # The triplet margin parameter
        M = 0.05

        # Calculate euclidean (not square rooted) distance between anchor image and positive text sample
        pos_dist_img = tf.square(tf.subtract(anchor_image, positive_text))
        pos_dist_img = tf.reduce_sum(pos_dist_img, 1)  # sum over all columns (dim 1)

        # Calculate euclidean (not square rooted) distance between anchor image and negative text sample
        neg_dist_img = tf.square(tf.subtract(anchor_image, negative_text))
        neg_dist_img = tf.reduce_sum(neg_dist_img, 1)  # sum over all columns (dim 1)

        # Calculate the loss over the image anchor
        basic_img_loss = tf.add(tf.subtract(pos_dist_img, neg_dist_img),
                                M)  # calculate the loss for each image-text triplet in the batch
        img_loss = tf.maximum(basic_img_loss, 0.0)

        return img_loss

    # Calculate the loss based on the text query anchor, compared with positive and negative image
    def text_loss(self, input):

        anchor_text, positive_image, negative_image = input

        # The triplet margin parameter
        M = 0.05

        # Calculate the euclidean distance from anchor text to positive image sample
        pos_dist_text = tf.square(tf.subtract(anchor_text, positive_image))
        pos_dist_text = tf.reduce_sum(pos_dist_text, 1)  # sum over all columns (dim 1)

        # Calculate the euclidean distance from anchor text to positive image sample
        neg_dist_text = tf.square(tf.subtract(anchor_text, negative_image))
        neg_dist_text = tf.reduce_sum(neg_dist_text, 1)  # sum over all columns (dim 1)

        # Calculate the loss over the text anchor
        basic_text_loss = tf.add(tf.subtract(pos_dist_text, neg_dist_text),
                                 M)  # calculate loss for each playground triplet in the batch
        text_loss = tf.maximum(basic_text_loss, 0.0)

        return text_loss

    # Calculation of the triplet loss over both image and text anchors
    def bi_directional_triplet_loss(self, input):
        img_loss, text_loss = input

        # Relative importance term
        lambda1 = 1
        if BI_DIRECTIONAL:
            lambda2 = 1.5
        else:
            lambda2 = 0

        mean_img_loss = tf.reduce_mean(img_loss, 0)  # calculate the mean loss over the image text triplet for the batch

        mean_text_loss = tf.reduce_mean(text_loss, 0) # calculate the mean loss over the playground triplet for the batch

        batch_loss = tf.add(tf.scalar_mul(lambda1, mean_img_loss), tf.scalar_mul(lambda2, mean_text_loss))

        return batch_loss

    ###### Creation of encoding models, image and text #####################

    # Creating the embedding for the images, which generates a vector that can be compared with the text query embedding
    def create_image_embedding_model(self):
        
        # Create VGG19-net from pre trained model, trained on imagenet, exclude the head of the model
        base_image_encoder = VGG19(weights='imagenet', include_top=False)
        image_input = base_image_encoder.input
        base_image_encoder = base_image_encoder.output

        # Use global average pooling to convert to a flat tensor
        base_image_encoder = GlobalAveragePooling2D(name='start_embedding_layers')(base_image_encoder)
        
        # Adding the embedding network on base image encoder
        dense_i1 = Dense(2048, activation='relu', name='dense_i1')(base_image_encoder)
        dropout_i1 = Dropout(0.5)(dense_i1)
        dense_i2 = Dense(512)(dropout_i1)

        # Batch normalize and L2 normalize output
        batch_norm = BatchNormalization()(dense_i2)
        l2_norm_i = Lambda(lambda x: backend.l2_normalize(x, axis=1))(batch_norm)

        img_embedding_model = Model(inputs=[image_input], outputs=[l2_norm_i], name='img_emb_model')

        # Freeze all the pre trained image-encoding layers (the vgg19 layers)
        for i, layer in enumerate(img_embedding_model.layers):
            if layer.name == 'dense_i1':
                break
            layer.trainable = False

        return img_embedding_model

    # Creating the embedding for the text query, which generates a vector which can be compared with the image embedding
    def create_text_embedding_model(self):
        base_text_encoder, text_input = self.create_lstm_query_embedding_layers()

        # Adding the embedding network for the base text encoder
        dense_t1 = Dense(1024, activation='relu')(base_text_encoder)
        dropout_t1 = Dropout(0.5)(dense_t1)
        dense_t2 = Dense(512)(dropout_t1)

        # Batch normalize and L2 normalize output
        batch_norm = BatchNormalization()(dense_t2)
        l2_norm_t = Lambda(lambda x: backend.l2_normalize(x, axis=1))(batch_norm)
        
        text_embedding_model = Model(inputs=[text_input], outputs=[l2_norm_t], name='text_emb_model')

        return text_embedding_model

    # Encodes the query using an LSTM
    def create_lstm_query_embedding_layers(self):

        text_input = Input(shape=(None,))

        word_embedding = Embedding(input_dim=self.vocab_size, #embedding_matrix.shape[0], # Vocabulary size
                                output_dim=WORD_EMBEDDING_SIZE, #embedding_matrix[1], # vector encoding size
                                weights=[self.word_embedding_weights], # Pre trained word embedding
                                input_length= 25,
                                trainable=False, # If weights can be modified through back propagation
                                mask_zero=True, # Clip away the zero embeddings
                                name='word_embedding')(text_input)
        
        # Debugging layer output
        #word_embedding = Lambda(self.print_layer)(word_embedding)
        
        lstm_1 = LSTM(512)(word_embedding)
        return lstm_1, text_input

    # Debug
    def print_layer(self, input):
        tensor = input

        tensor = tf_print(tensor, [tensor], 'anchor_image: ')

        return tensor

    ###### End of creation of encoding models, image and text #####################

if __name__ == '__main__':
    model_class = triplet_loss_embedding_graph()
    model = model_class.model
    model.summary()