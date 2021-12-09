from os.path import exists
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import json
import random
from joblib import dump, load
import utils
import lda_model
import lda_tuner


dataset_dir = 'data/dataset.csv'

def find_similar_texts(text):
    corpus = utils.load_job_description(dataset_dir)
    search_params = {'n_components': [2, 4, 5, 10, 15, 20]}
    n_features = 10000
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=n_features)
    if not exists(utils.MODEL_DIR) or not exists(utils.VECTOR_DIR) or not exists(utils.OUTPUT_DIR):
        lda_tuner.find_best_model(dataset_dir, search_params, vectorizer=tf_vectorizer)
    best_model = load(utils.MODEL_DIR)
    lda_output = load(utils.OUTPUT_DIR)
    tf_vectorizer = load(utils.VECTOR_DIR)
    doc_ids, docs = lda_model.similar_documents(text=text, vectorizer=tf_vectorizer, model=best_model, doc_topic_probs    =lda_output, documents = corpus, top_n=50)
    return doc_ids, docs
