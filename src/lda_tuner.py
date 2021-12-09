from joblib import dump
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import json
import random
import utils


def find_best_model(dataset_dir, search_params, vectorizer):
    corpus = utils.load_job_description(dataset_dir)
    vectorized_data = vectorizer.fit_transform(corpus)
    lda= LatentDirichletAllocation(random_state=1)
    model = GridSearchCV(lda, param_grid=search_params)
    model.fit(vectorized_data)
    best_model = model.best_estimator_
    print("Best Model's Params: ", model.best_params_)
    lda_output = best_model.transform(vectorized_data)
    print('Saving vector, model, and output')
    dump(vectorizer, utils.VECTOR_DIR)
    dump(best_model, utils.MODEL_DIR)
    dump(lda_output, utils.OUTPUT_DIR)
