import nltk
from nltk import pos_tag, RegexpParser, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.tree import Tree
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import state_union
import fasttext

class NamedEntityRecognition:
    def __init__(self):
        self.sentTokenizer = PunktSentenceTokenizer()
        self.max_sentence_length = 50
        self.skipgram_model = fasttext.load_model('skipgram.bin')

    def preprocess_lstm(self, text):
        pass

    def preprocess_nltk(self, text):
        sentences = self.sentTokenizer.tokenize(text)
        pos_tags = []
        for sent in sentences:
            pos_tags += pos_tag(word_tokenize(sent))

        return pos_tags

    def NLTK(self, text):
        pos_tagged = self.preprocess_nltk(text)
        chunked = ne_chunk(pos_tagged)
        continuous_chunk = []
        current_chunk = []

        for chunk in chunked:
            if hasattr(chunk, 'label'):
                label = chunk.label()
            if type(chunk) == Tree:
                current_chunk.append(" ".join([token for token, pos in chunk.leaves()]))
            elif current_chunk:
                named_entity = ' '.join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append('{} {}'.format(label, named_entity))
                    current_chunk = []
            else:
                continue

        return continuous_chunk
