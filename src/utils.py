from csv import DictReader
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def load_job_description(dataset):
    with open(dataset, newline='', encoding='utf-8') as f:
        reader = DictReader(f)
        return [line['jobDescription'].encode('utf-8') for line in reader]


def predict_topic(vectorizer, model, text):
    vectorized_text = vectorizer.transform(text)
    return model.transform(vectorized_text)

