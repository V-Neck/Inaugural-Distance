import nltk
from collections import Counter

class Corpus:

    def __init__(self, corpus_text, doc_name, n_gram_length):
        self.n_gram_length = n_gram_length + 1
        tokens = nltk.word_tokenize(corpus_text)
        self.n_gram_count = Counter()
        self.doc_length = len(tokens)
        self.doc_name = doc_name


        for i in range(0, self.doc_length - self.n_gram_length):
            self.n_gram_count[tokens[i]] += 1
            for j in self.index_range(i+1, i+self.n_gram_length):
                n_gram = " ".join([x for x in tokens[i:j]])
                self.n_gram_count[n_gram] += 1


    def index_range(self, start, stop):
        if start > self.doc_length:
            return []
        if stop > self.doc_length:
            return range(start, self.doc_length)
        return range(start, stop)
