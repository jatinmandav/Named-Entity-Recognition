from tqdm import tqdm
import numpy as np
import os
import pickle
from nltk import pos_tag, RegexpParser, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.tree import Tree
import fasttext

class Preprocess:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        with open(self.file_path, 'r') as file_:
            self.data = file_.read()

    def clean(self):
        unwanted_token = '-DOCSTART-'
        clean_data = []
        for line in self.data.split('\n'):
            if unwanted_token in line:
                pass
            else:
                clean_data.append(line)

        self.data = '\n'.join(clean_data)

    def part_of_speech(self):
        self.pos_tags = []
        for sent in self.sentences:
            self.pos_tags += pos_tag(sent)

        self.pos_dict = {}

        for pos in self.pos_tags:
            self.pos_dict[pos[0]] = pos[1]


    def get_max_length(self):
        data = self.data.split('\n')
        self.sentences = []
        self.labels = []

        sent_lenghts = []
        i = 0
        sent = []
        label = []
        for line in data:
            if line != '\n' and line != '':
                sent.append(line.split(' ')[0])
                label.append(line.split(' ')[1])
            if len(line) == 0:
                if sent != []:
                    self.sentences.append(sent)
                    self.labels.append(label)
                    sent = []
                    label = []
                sent_lenghts.append(i)
                i = 0
            else:
                i += 1

        return max(sent_lenghts)

    def load_emb_model(self, path='skipgram.bin'):
        self.model = fasttext.load_model(path)

    def get_pos_cap_embedding(self, word):
        one_hot = np.zeros(6)
        tag = self.pos_dict[word]
        if tag in ['NN', 'NNS']:
            one_hot[0] = 1
        elif tag == 'FW':
            one_hot[1] = 1
        elif tag in ['NNP', 'NNPS']:
            one_hot[2] = 1
        elif 'VB' in tag:
            one_hot[3] = 1
        else:
            one_hot[4] = 1

        if ord('A') <= ord(word[0]) <= ord('Z'):
            one_hot[5] = 1

        return one_hot

    def generate_training_dataset(self):
        max_length = self.get_max_length()
        embeddings = []

        label_one_hot = []

        for labels, sent in zip(self.labels, self.sentences):
            sent_embedding = []
            sent_label = []
            for label, word in zip(labels, sent):
                embed = list(self.model[word])
                embed += list(self.get_pos_cap_embedding(word))
                sent_embedding.append(embed)

                if label.endswith('PER'):
                    sent_label.append(np.array([1, 0, 0, 0, 0]))
                elif label.endswith('LOC'):
                    sent_label.append(np.array([0, 1, 0, 0, 0]))
                elif label.endswith('ORG'):
                    sent_label.append(np.array([0, 0, 1, 0, 0]))
                elif label.endswith('MISC'):
                    sent_label.append(np.array([0, 0, 0, 1, 0]))
                elif label.endswith('O'):
                    sent_label.append(np.array([0, 0, 0, 0, 1]))
                else:
                    print('ERROR', label)

            padding = max_length - len(sent_embedding)
            for _ in range(padding):
                sent_embedding.append(np.zeros(100+6))
                sent_label.append(np.array([0]*5))

            embeddings.append(sent_embedding)
            label_one_hot.append(sent_label)

        print(np.array(embeddings).shape)
        print(np.array(label_one_hot).shape)

        with open('training_data.pkl', 'wb') as file_:
            pickle.dump(embeddings, file_)

        with open('training_label.pkl', 'wb') as file_:
            pickle.dump(label_one_hot, file_)

preprocess = Preprocess('dataset.txt')
preprocess.read()
preprocess.clean()
preprocess.get_max_length()
preprocess.part_of_speech()
preprocess.load_emb_model()
preprocess.generate_training_dataset()
