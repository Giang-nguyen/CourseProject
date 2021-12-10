import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import utils
from joblib import dump, load


def similar_documents(text, vectorizer, model, doc_topic_probs, documents, top_n=5):
    topics = utils.predict_topic(vectorizer, model, [text])
    dists = euclidean_distances(topics.reshape(1, -1), doc_topic_probs)[0]
    doc_ids = np.argsort(dists)[:top_n]
    return doc_ids, np.take(documents, doc_ids)
