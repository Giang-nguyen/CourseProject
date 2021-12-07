import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import json
import random
import utils
import lda_model
import lda_tuner


n_components = 10
n_top_words = 20
dataset_dir = 'data/dataset.csv'
corpus = utils.load_job_description(dataset_dir)
mytext = [corpus[125]]
corpus.pop(125)
search_params = {'n_components': [2, 4, 10, 20]}
n_features = 1000
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=n_features)

# Uncomment the below lines for running grid search algorithm to find best params for the model
#best_model, lda_output = lda_tuner.find_best_model(dataset_dir, search_params, vectorizer=tf_vectorizer)
#doc_ids, docs = lda_model.similar_documents(text=mytext, vectorizer=tf_vectorizer, model=best_model, doc_topic_probs    =lda_output, documents = corpus, top_n=5)

best_model, lda_output = lda_model.train_model(data=corpus, vectorizer=tf_vectorizer, random_state=1, n_components=4)
doc_ids, docs = lda_model.similar_documents(text=mytext, vectorizer=tf_vectorizer, model=best_model, doc_topic_probs    =lda_output, documents = corpus, top_n=5)

print('text')
print(mytext)
print()
print('docs')
for doc in docs:
    print('doc')
    print(doc)
    print()