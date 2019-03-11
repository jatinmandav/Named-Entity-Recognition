import pickle
import fasttext

import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class GetSentence(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s:[(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]

        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

class Preprocess:
    def __init__(self):
        self.max_len = 50

    def read_data(self):
        data = pd.read_csv('ner_dataset.csv', sep=',', encoding='latin1')
        data = data.fillna(method='ffill')

        words = list(set(data["Word"].values))
        words.append('ENDPAD')

        n_words = len(words)

        self.tags = list(set(data['Tag'].values))
        n_tags = len(self.tags)

        sent_extractor = GetSentence(data)
        self.sentences = sent_extractor.sentences

        raw_sentences = []
        for sent in self.sentences:
            s = []
            for word in sent:
                s.append(word[0])

            raw_sentences.append(' '.join(s))

    def process(self):
        self.X = [[word[0] for word in sent] for sent in self.sentences]

        padded_X = []
        for seq in self.X:
            new_seq = []
            for i in range(self.max_len):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append('__PAD__')

            padded_X.append(new_seq)

        self.X = padded_X[:]
        print(self.X[3])

        self.tag2index = {t: i for i, t in enumerate(self.tags)}
        print(self.tag2index)

        self.Y = [[self.tag2index[word[2]] for word in sent] for sent in self.sentences]
        self.Y = pad_sequences(maxlen=self.max_len, sequences=self.Y, padding='post', value=self.tag2index['O'])

        print(self.tag2index, len(self.tag2index))

        one_hot_y = []
        for sent in self.Y:
            sent_one_hot = []
            for word in sent:
                one_hot = np.zeros(len(self.tag2index))
                one_hot[word] = 1
                sent_one_hot.append(one_hot)
            one_hot_y.append(sent_one_hot)

        self.Y = np.array(one_hot_y)

        print(np.array(one_hot_y).shape)
        print(one_hot_y[0])

        with open('tag2index.pkl', 'wb') as file_:
            array = np.array(list(self.tag2index.keys()))
            pickle.dump(array, file_)


    def split_train_test(self):
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.X, self.Y, test_size=0.1, shuffle=True)

        self.trainX = np.array(self.trainX)
        self.trainY = np.array(self.trainY)
        self.testX = np.array(self.testX)
        self.testY = np.array(self.testY)

        print(self.trainX.shape, self.trainY.shape)
        print(self.testX.shape, self.testY.shape)

        pickle.dump(self.trainX, open('train_x.pkl', 'wb'))
        pickle.dump(self.trainY, open('train_y.pkl', 'wb'))
        pickle.dump(self.testX, open('test_x.pkl', 'wb'))
        pickle.dump(self.testY, open('test_y.pkl', 'wb'))


preprocessor = Preprocess()
preprocessor.read_data()
preprocessor.process()
preprocessor.split_train_test()
