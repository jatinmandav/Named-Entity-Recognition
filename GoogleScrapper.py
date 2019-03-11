from googlesearch import search
from urllib.request import urlopen
import nltk

from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request

from tqdm import tqdm

class GoogleScrapper:
    def __init__(self, no_results):
        self.no_results = no_results
        self.results = []

    def search(self, query):
        self.results = search(query, stop=self.no_results)

    def tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(self, body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)
        return u" ".join(t.strip() for t in visible_texts)

    def get_text(self):
        self.text_body = []
        with tqdm(total=self.no_results) as pbar:
            for url in self.results:
                try:
                    file = urlopen(url)
                    content = file.read()
                    content = self.text_from_html(content)
                    self.text_body.append({'url': url, 'text':content})
                except Exception as e:
                    print(e)

                pbar.update(1)
        return self.text_body

scrapper = GoogleScrapper(1)
scrapper.search('Barack Obama United States Democratic Group')
print(scrapper.get_text()[:1000])
