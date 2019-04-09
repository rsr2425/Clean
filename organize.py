'''
Module: Organize

This is the main file of the program "Clean" which should be used to call
the tool from the command line.

NOTE: Although the final form of this application will actually move the
files into folders based on the categories (and eventually hierarchy)
developed eventually, for the time being it just prints out the proposed
structure.

Low-hanging fruit for improvement:
- Use argparse instead of sys.argv directly
- I should stop being lazy about dealing with paths because it's making
  the code a bit fragile.
- Might want to profile the code soon to scale up because it's a bit lagely
  on large files.

'''

import os
import sys
import spacy
import pickle

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
     try:
         text = f.read()
         docs[filename] = nlp(text).vector
         docs[filename] = np.array(docs[filename]).reshape(-1, 1)
         docs[filename] = np.linalg.norm(docs[filename], axis=1)
         docs[filename] = np.squeeze(docs[filename])
     except UnicodeDecodeError:
         pass
     f.close()
n = len(docs)

# determine number of clusters
X = list(docs.values())
n_clusters = int(sys.argv[2])
#print(docs)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
results = list(zip(docs.keys(), kmeans.labels_))

# count how many of each class was assigned to a cluster with cls_cnt
cls_cnt = [None] * n_clusters
for r in results:
     # print(r)
     idx = r[1]
     class_name = r[0].split(sep='_')[0]
     try:
         cls_cnt[idx][class_name] += 1
     except KeyError:
         cls_cnt[idx][class_name] = 1
     except TypeError:
         cls_cnt[idx] = {}
         cls_cnt[idx][class_name] = 1

# for i, c in enumerate(cls_in_clstr):
#    print(f'Cluster {i}:')
#    print(f'{c}')
for i, cs in enumerate(cls_cnt):
     print('*'*20)
     print(f'Cluster {i}: {cs}')
     res = 0
     for key, c in cs.items():
         print(f'{key}: {c}')
         res += c
     print(res)

# store for analysis later in pickle file
# e.g. in 'All_clusters_10k.pkl'
pklf = open(sys.argv[3], 'wb')
pickle.dump(cls_cnt, pklf)
pklf.close()