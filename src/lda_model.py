import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import json
import random
import utils


def similar_documents(text, vectorizer, model, doc_topic_probs, documents, top_n=5):
    topics = utils.predict_topic(vectorizer, model, text)
    dists = euclidean_distances(topics.reshape(1, -1), doc_topic_probs)[0]
    doc_ids = np.argsort(dists)[:top_n]
    return doc_ids, np.take(documents, doc_ids)


def train_model(data, vectorizer, n_components, random_state):
    vectorized_data = vectorizer.fit_transform(data)
    model = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    model.fit(vectorized_data)
    print("Model's:")
    print(model)
    output = model.transform(vectorized_data)
    return model, output
    