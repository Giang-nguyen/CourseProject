import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import re
import spacy


def load_job_description(dataset):
    jobs = pd.read_csv(dataset, encoding='utf-8')
    return jobs['jobDescription'].drop_duplicates().dropna().apply(lambda x: clean(x)).to_list()


def clean(text):
    # Remove URL
    new_text = re.sub(r'http\S+', '', text)
    # Remove numbers
    new_text = re.sub(r'[0-9]+', '', new_text)
    #new_text = lemmatization(new_text)
    return new_text


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


# function to lemmatize text
def lemmatization(text):
    s = [token.lemma_ for token in nlp(text)]
    return ' '.join(s)


def predict_topic(vectorizer, model, text):
    vectorized_text = vectorizer.transform(text)
    return model.transform(vectorized_text)

ROOT = 'model'
MODEL_DIR = os.path.join(ROOT, 'best_model.joblib')
OUTPUT_DIR = os.path.join(ROOT, 'output.joblib')
VECTOR_DIR = os.path.join(ROOT, 'vector.joblib')