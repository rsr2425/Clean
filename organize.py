'''
Module: Organize

This is the main file of the program "Clean" which should be used to call
the tool from the command line.

NOTE: Although the final form of this application will actually move the
files into folders based on the categories (and eventually hierarchy)
developed eventually, for the time being it just prints out the proposed
structure.
'''

import os
import sys
import spacy

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

# load up spacy and ready it for use
nlp = spacy.load('en_core_web_lg')

# load directory intended for processing
dir = sys.argv[1]

# scan over all files and create doc2vec representation
docs = {}
for file in os.listdir(dir):
     filename = os.fsdecode(file)
     f =  open(dir + '/' + file)
     text = f.read()
     docs[filename] = nlp(text).vector
     #docs[filename] = normalize(np.array(docs[filename]), axis=0)
n = len(docs)


# determine number of clusters
X = list(docs.values())
n_clusters = int(sys.argv[2])

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
print(list(zip(docs.keys(), kmeans.labels_)))

# create a label for each cluster
