# Named-Entity-Recognition

Named Entity Recognition done using NLTK library and LSTM based network.

`main.py` takes three input, Name of Person, Name of Organization, Location and calls GoogleScrapper() file to get results from google search and passes the text to **NLTK based NER**. Joining of LSTM based NER with GoogleSearch is to be done soon.

## Usage - NLTK Based

For NLTK based NER, run `main.py` using terminal

`python3 main.py --method=NLTK --per=NAME_OF_PERSON["FirstName LastName"] --org=NAME_OF_ORG --loc=NAME_OF_LOCATION --result=NUMBER_OF_WEBPAGES_TO_PROCESS`

For more details:

`python3 main.py --help`

### Algorithm

 - Preprocessing - Converting each sentence to a list of words.
 - POS-Tagging - Using NLTK's `pos_tag` to tag part-of-speech of each word.
 - Chunking - Using NLTK's `ne_chunk` to form chunks and then forming continous chunks by combining words under same label on same branch. Eg: "PERSON Barack", "PERSON Obama" is converted to "PERSON Barack Obama"
 - Final Result: The list of these NER's is returned.

## Usage - LSTM Based

Download `logs/`, `tag2index.pkl`, `skipgram.bin` and `skipgram.vec` from  Releases section of the repo and place inside this repo.

For LSTM based NER, run `predict_lstm.py` using terminal

`python3 predict_lstm.py text.txt`

where `text.txt` contains raw text to tag. Output prints each word tagged with entity.

### Training

  - The model is trained on `ner_dataset.csv` [Source: Kaggle].
  - Each sentence is padded by padding token `__PAD__` to a fixed max_len of 50.
  - Word embeddings of each word is generated using `skipgram` model from `fasttext`.
  - Label of each word is the `Tag` from `ner_dataset.csv` ['B-art' 'B-nat' 'I-nat' 'B-eve' 'O' 'B-gpe' 'I-tim' 'B-org' 'I-org' 'B-tim' 'I-art' 'I-per' 'B-per' 'I-gpe' 'I-geo' 'I-eve' 'B-geo']
  
  - Training on custom dataset with complete tutorial to be released soon. Until then feel free to play with code.
#### Model:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Input (InputLayer)           (None, 50, 100)           0         
_________________________________________________________________
LSTM-1 (LSTM)                (None, 256)               365568    
_________________________________________________________________
RepeactVector-1 (RepeatVecto (None, 50, 256)           0         
_________________________________________________________________
LSTM-2 (LSTM)                (None, 50, 256)           525312    
_________________________________________________________________
time_distributed_1 (TimeDist (None, 50, 17)            4369      
_________________________________________________________________
activation_1 (Activation)    (None, 50, 17)            0         
=================================================================
Total params: 895,249
Trainable params: 895,249
Non-trainable params: 0
_________________________________________________________________

```

The model was trained for 50 epochs. 

Accuracy achieved 98.5%


