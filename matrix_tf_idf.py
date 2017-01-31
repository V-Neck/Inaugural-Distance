from Corpus import Corpus
from Document import Document
import io, os
from AsciiDammit import asciiDammit
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.spatial import distance
from sklearn.preprocessing import normalize

#1-grams through n-grams will be compilated
n_gram_length = 4
corpus_title = "inaugural/all.txt"
corp_text = asciiDammit(open(corpus_title).read())
corp = Corpus(corp_text, corpus_title, n_gram_length)

#Just a mapping of the vocabulary to the natural numbers
vocab_map = {}

for index, key in enumerate(corp.n_gram_count):
     vocab_map[key] = index

#List of inaugural speech document objects
speeches = []

#index 0 is .DS_STore
folder = "inaugural"
for file in os.listdir(folder)[1:]:
    doc_name = folder + "/" + file
    doc_text = open(doc_name, "r").read()
    doc_text = asciiDammit(doc_text)
    speeches.append(Document(doc_text, file, corp, n_gram_length))

#Document by words
m = len(speeches)
n = len(vocab_map.keys())

#An m by n matrix, in which the columns are the vocabulary, and the rows are the various speeches. Since the speeches contain a fair amount of variety, most of the rows will be filled with zeros
#Would use column sparse matrix, but generates an error in the following loop
count_matrix = lil_matrix((m, n))

doc_indices = [i for i in enumerate(speeches)]
for doc_idx, speech in enumerate(speeches):
    for word, count in speech.tf_idf.items():
        word_idx = vocab_map[word]
        count_matrix[doc_idx, word_idx] = count

#Normalizes the matrix by columns, to values between 0 and 1
norm_matrix = normalize(count_matrix, norm = 'l1', axis = 0)
norm_matrix = csc_matrix(norm_matrix)

#cosine distance from corpus document
dist_from_norm = []
for i in range(0,norm_matrix.shape[0]):
    dist_from_norm.append( [doc_indices[i][1].doc_name, distance.cosine(norm_matrix[i].toarray(), norm_matrix[-1].toarray())])

for x in sorted(dist_from_norm, key=lambda tup: tup[1]):
    print x

#TODO Consider morphosyntantic structure
    #Figure out how to utilize CSC matrix
    #Generalize generation of "corpus"
    #Consider moving reading to object
