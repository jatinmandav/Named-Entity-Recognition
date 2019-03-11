from keras.models import Model
import numpy as np
import pickle
import fasttext
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation
from sklearn.model_selection import train_test_split
from keras.layers import Dense, TimeDistributed, RepeatVector, Input, Lambda
from keras.backend import tf
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data', help='Pickle file of data', required=True)
parser.add_argument('--labels', help='Pickle file of labels', required=True)
parser.add_argument('--logs', help='Path to logs', default='logs/')

args = parser.parse_args()

class ModelDef:
    def __init__(self, data_path, label_path):
        self.hidden_size = 256
        self.max_len_sentence = 50
        tagDict = pickle.load(open('tag2index.pkl', 'rb'))
        print(tagDict)
        self.n_tags = len(tagDict)
        self.batch_size = 32
        self.epochs=50

        self.read_data(data_path, label_path)
        self.load_skipgram()

    def read_data(self, data_path, label_path):
        self.embeddings = np.array(pickle.load(open(data_path, 'rb')))
        self.labels = np.array(pickle.load(open(label_path, 'rb')))

        #self.labels = self.labels.reshape(self.labels.shape[0], self.labels.shape[1], 1)

        print(self.embeddings.shape, self.labels.shape)

    def load_skipgram(self, path='skipgram.bin'):
        self.skipgram_model = fasttext.load_model(path)

    def get_embeddings(self, x):
        embeddings = []
        for word in x:
            embeddings.append(self.skipgram_model[word])

        return np.array(embeddings)

    def generator(self):
        while True:
            no_batches = int(len(self.embeddings)/self.batch_size)

            for i in range(no_batches):
                start_index = i*self.batch_size
                sent_embedding = []
                output = []

                for j in range(self.batch_size):
                    try:
                        sent_embedding.append(self.get_embeddings(self.embeddings[start_index+j]))
                        output.append(self.labels[start_index+j])
                    except Exception as e:
                        print(e)

                yield np.array(sent_embedding), np.array(output)

    def build(self, shape=(50, 100)):
        inp = Input(shape=shape, name='Input')

        x = LSTM(self.hidden_size, name='LSTM-1', return_sequences=False)(inp)
        x = RepeatVector(self.max_len_sentence, name='RepeactVector-1')(x)
        x = LSTM(self.hidden_size, name='LSTM-2', return_sequences=True)(x)

        x = TimeDistributed(Dense(self.n_tags))(x)
        x = Activation('softmax')(x)

        self.model = Model(inputs=inp, outputs=x)


    def compile(self, opt):
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def fit(self):
        generator_ = self.generator()
        logging = TensorBoard(log_dir=log_dir)
        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}.h5',
            monitor='acc', save_weights_only=True, save_best_only=True, period=3)
        reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='acc', min_delta=0, patience=10, verbose=1)

        self.compile(Adam(lr=1e-3))

        self.model.fit_generator(generator=generator_, steps_per_epoch=int(len(self.embeddings)/self.batch_size),
                        epochs=self.epochs, callbacks=[logging, checkpoint, reduce_lr, early_stopping])


        self.model.save(log_dir + 'stage-1-model.h5')

        self.compile(Adam(lr=1e-4))
        
        self.model.save(log_dir + 'final-model.h5')


log_dir = args.logs
data_path = args.data
label_path = args.labels

model = ModelDef(data_path, label_path)
model.build()
model.summary()
model.fit()
