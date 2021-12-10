import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import re
import spacy


def load_job_description(dataset):
    df_data = pd.read_csv(dataset, encoding='utf-8')
    return df_data['jobDescription'].drop_duplicates().dropna().apply(lambda x: clean(x)).to_list()


def load_skill_description(dataset):
    df_data = pd.read_csv(dataset, encoding='utf-8')
    return df_data.loc[:, ['title', 'jobSkill']].drop_duplicates().dropna().apply(lambda row: clean_skill_descriptions(row), axis=1)


def clean_skill_descriptions(row):
    row['title'] = clean(row['title'])
    return row



def clean(text):
    # Remove URL
    new_text = re.sub(r'http\S+', '', text)
    # Remove numbers
    new_text = re.sub(r'[0-9]+', '', new_text)
    #new_text = lemmatization(new_text)
    return new_text.encode('utf-8')


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
SKILL_OUTPUT_DIR = os.path.join(ROOT, 'skill_description.joblib')
