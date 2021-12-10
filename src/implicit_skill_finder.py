from os.path import exists
from numpy.lib.function_base import vectorize
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from joblib import dump, load
import utils
import lda_model
import lda_tuner


dataset_dir = 'data/dataset.csv'
skill_description_dir = 'data/Skill Description.csv'
corpus = utils.load_job_description(dataset_dir)
skill_descriptions = utils.load_skill_description(skill_description_dir)

def train_model():
    search_params = {'n_components': [2, 4, 5, 10, 15, 20]}
    n_features = 10000
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=n_features)
    if not exists(utils.MODEL_DIR) or not exists(utils.VECTOR_DIR) or not exists(utils.OUTPUT_DIR):
        lda_tuner.find_best_model(dataset_dir, search_params, vectorizer=tf_vectorizer)


def find_similar_texts(text, dataset, doc_topic_probs, n_result):
    train_model()
    best_model = load(utils.MODEL_DIR)
    tf_vectorizer = load(utils.VECTOR_DIR)
    doc_ids, docs = lda_model.similar_documents(text=text, vectorizer=tf_vectorizer, model=best_model, doc_topic_probs = doc_topic_probs, documents = dataset, top_n=n_result)
    return doc_ids, docs

def load_skill_description_lda():
    if not exists(utils.MODEL_DIR) or not exists(utils.VECTOR_DIR):
        return np.empty((0, 0))
    
    if not exists(utils.SKILL_OUTPUT_DIR):
        best_model = load(utils.MODEL_DIR)
        tf_vectorizer = load(utils.VECTOR_DIR)
        vectorized_skill_descriptions = tf_vectorizer.transform(skill_descriptions['title'])
        output = best_model.transform(vectorized_skill_descriptions)
        dump(output, utils.SKILL_OUTPUT_DIR)
        return output
    else:
        return load(utils.SKILL_OUTPUT_DIR)


def find_similar_docs(docs, n_result):
    return find_similar_texts(docs, corpus, load(utils.OUTPUT_DIR), n_result)


def find_related_skills(docs, n_result):
    skill_ids, _ = find_similar_texts(docs, skill_descriptions['title'], load_skill_description_lda(), n_result)
    return skill_descriptions.iloc[skill_ids, :]['jobSkill'].apply(lambda x: x.strip())

