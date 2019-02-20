from keras.callbacks import Callback
from evaluate_acc import evaluate_acc

class val_accuracy(Callback):
    def __init__(self, tokenizer, image_embedding_model, text_embedding_model, verbose=1):
        self.tokenizer = tokenizer
        self.image_embedding_model = image_embedding_model
        self.text_embedding_model = text_embedding_model
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        acc_evaluator = evaluate_acc(self.tokenizer, self.image_embedding_model, self.text_embedding_model)
        top_1, top_5, top_10 = acc_evaluator.top_1_5_10_accuracy()
        # Add the rank acc to the logs dict so it can be used for other callback functions to evaluate on
        logs['r1_acc'] = top_1
        logs['r5_acc'] = top_5
        logs['r10_acc'] = top_10
        if self.verbose == 1:
            print('Val top_1 acc: {}, val top_5 acc: {}, Val top_10 acc: {}'.format(top_1, top_5, top_10))
