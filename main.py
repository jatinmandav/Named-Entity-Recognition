import argparse
from NamedEntityRecognition import NamedEntityRecognition
from GoogleScrapper import GoogleScrapper
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument('--method', help='NLTK or LSTM', required=True)
parser.add_argument('--per', help='Name of a person', required=True)
parser.add_argument('--org', help='Name of the Organization', required=True)
parser.add_argument('--loc', help='Location', required=True)
parser.add_argument('--results', help='Number of Results', default=3, type=int)

args = parser.parse_args()

query = args.per + ' ' + args.loc + ' ' + args.org

scrapper = GoogleScrapper(args.results)
print('Extracting Text ..')
scrapper.search(query)
results = scrapper.get_text()
print('Done.')

query_list = ['PERSON {}'.format(args.per), 'ORGANIZATION {}'.format(args.per), 'LOCATION {}'.format(args.per)]

ner_tagger = NamedEntityRecognition()

print('Tagging Named Entities ..')
for dict_ in results:
    result = ner_tagger.NLTK(dict_['text'])
    
    count = Counter(result)
    per_count = count[query_list[0]]
    org_count = count[query_list[1]]
    loc_count = count[query_list[2]]

    if (per_count > 0) or (org_count > 0) or (loc_count > 0):
        print('URL: {}'.format(dict_['url']))

        print('Occurance of {}[{}]: {}'.format(args.per, 'PERSON', per_count))
        print('Occurance of {}[{}]: {}'.format(args.org, 'ORGANIZATION', org_count))
        print('Occurance of {}[{}]: {}'.format(args.loc, 'LOCATION', loc_count))
        print()

print('Done.')

print(result)
