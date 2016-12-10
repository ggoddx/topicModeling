from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import nltk

from nltk.corpus import stopwords

from sklearn.decomposition import NMF, LatentDirichletAllocation

# getting data from BBC news

from bs4 import BeautifulSoup
import os
from os.path import join, getsize


import string




documents=[]
labels=[]
document_id=[]

print("Loading ap dataset...")
t0 = time()

for subdir, dirs, files in os.walk('bbc'):
	for name in files:
		labels.append(subdir.split('/')[0])
		#id is the path to the file
		id=join(subdir, name)
		document_id.append(id)

		with open (id, "r") as myfile:
   			data=myfile.read()
			documents.append(data)

print("done in %0.3fs." % (time() - t0))
print (len(documents), " BBC news documents are loaded")

pstemmer = nltk.PorterStemmer()






n_samples = 2000
n_features = 1000
n_topics = 5
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


data_samples=documents


# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   decode_error='replace', max_features=n_features, strip_accents='unicode', stop_words='english')



t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))



# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,decode_error='replace', strip_accents='unicode',
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))


lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)











#removing punctuation 



#remove stop words


	#
# stemming

	

# plot works distribution 

#fd = nltk.FreqDist(data)
#fd.plot()
#fd.plot(50, cumulative=True)
#fd.most_common(12)


