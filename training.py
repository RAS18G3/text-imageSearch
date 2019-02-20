from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from time import time
import pickle

from model_improve import triplet_loss_embedding_graph
from data_generator import Data_generator
from Val_accuracy import val_accuracy

#import tensorflow as tf

#run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#sess.run(op, feed_dict=fdict, options=run_options)

BATCH_SIZE = 100
EPOCHS = 100
PRE_TRAINED = False

class training:

    def __init__(self):

        # Create the batch data generator and get otger required information of the data
        self.training_generator = Data_generator(BATCH_SIZE, pre_trained=PRE_TRAINED)
        self.tokenizer = self.training_generator.tokenizer
        self.vocab_size = self.training_generator.vocabulary_size
        # The weights (glove) for the word embedding matrix used in model
        embedding_weights = self.training_generator.get_embedding_matrix()
        print('now printing embedding_weights')
        print(embedding_weights.shape)

        # The embedding model class
        self.TLEG = triplet_loss_embedding_graph(vocab_size=self.vocab_size, embedding_weights=embedding_weights)
        # The playground embedding triplet loss model
        self.model = self.TLEG.model
        if PRE_TRAINED:
            # Load pre-trained weights
            self.model.load_weights('./model/best_weights_r10.hdf5')

        self.batch_predict_model = self.TLEG.batch_prediction_model

        # Set up the valuation models
        self.text_embedding_model = self.TLEG.text_embedding_model
        self.image_embedding_model = self.TLEG.img_embedding_model

        # Function defining how the model should be trained
    def train(self, epochs):
        # Save the tokenizer to file for use at inference time
        with open('./model/tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ## For val_acc
        # Set up the accuracy evaluation callback object
        val_acc_callback = val_accuracy(self.tokenizer, self.image_embedding_model, self.text_embedding_model)

        # define a tensorboard callback
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        # Save the best model based on r1
        file_path_r1 = './model/best_weights_r1.hdf5'
        save_best_r1 = ModelCheckpoint(file_path_r1, monitor='r1_acc', verbose=0, save_best_only=True, mode='min')

        # Save the best model based on r5
        file_path_r5 = './model/best_weights_r5.hdf5'
        save_best_r5 = ModelCheckpoint(file_path_r5, monitor='r5_acc', verbose=0, save_best_only=True, mode='min')

        # Save the best model based on r10
        file_path_r10 = './model/best_weights_r10.hdf5'
        save_best_r10 = ModelCheckpoint(file_path_r10, monitor='r10_acc', verbose=0, save_best_only=True, mode='min')

        callbacks = [val_acc_callback, save_best_r1, save_best_r5, save_best_r10, tensorboard]
        
        self.model.fit_generator(generator=self.training_generator.improved_batch_generator(self.batch_predict_model),
                                 steps_per_epoch=self.training_generator.batch_per_epoch,
                                 epochs=epochs, verbose=2, callbacks=callbacks
                                )

if __name__ == '__main__':
    Trainer = training()
    Trainer.train(EPOCHS)



    
