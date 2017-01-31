from Corpus import Corpus
from math import log

class Document(Corpus):
    def __init__(self, file_text, doc_name, corpus, n_gram_length):
        Corpus.__init__(self, file_text, doc_name, n_gram_length)
        self.tf_idf = {}

        for n_gram in self.n_gram_count:
            tf = float(self.n_gram_count[n_gram])
            idf = float(self.doc_length * (corpus.n_gram_count[n_gram] + 1))
            self.tf_idf[n_gram] = log(tf/idf)
            # or binary
