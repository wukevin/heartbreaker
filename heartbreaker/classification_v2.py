import os
import sys
import multiprocessing
import logging

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import data_loader
import util
import plotting

# Used below links as guidance for how to do multiple metric evaluation + K-fold CV
# https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
# https://stackoverflow.com/questions/46598301/how-to-compute-precision-recall-and-f1-score-of-an-imbalanced-dataset-for-k-fold

# and below links for guidance on how combine with scaling step in an sklearn pipeline
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#examples-using-sklearn-pipeline-pipeline
# https://stackoverflow.com/questions/44446501/how-to-standardize-data-with-sklearns-cross-val-score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

seed = 754927

def get_gscv(model, grid_params, scale=False, verbose=10):
    pipeline_steps = []

    if (scale):
        scaling_step = ('transformer', StandardScaler())
        pipeline_steps.append(scaling_step)
    
    feature_selection_step = ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear')))
    pipeline_steps.append(feature_selection_step)
    
    estimation_step = ('estimator', model)
    pipeline_steps.append(estimation_step)
    
    pipeline = Pipeline(pipeline_steps)
    
    cv = GridSearchCV(pipeline, grid_params, scoring=['f1', 'accuracy', 'precision', 'recall'], cv=10, verbose=verbose, refit='f1',  n_jobs=4)
    
    return cv

def adjust_params(params):
    return {'estimator__' + k: v for k, v in params.items()}

def main():
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates = util.continuous_to_categorical(rates, percentile_cutoff=75)
    x = data
    y = rates
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=seed)

    models = []
    
    dt = DecisionTreeClassifier()
    
    dt_params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 25, 50, 100],
        "min_samples_leaf": [0.001, 0.01, 0.1],
        "max_features": ["sqrt", "log2", None]
    }
    
    models.append((dt, dt_params, False))
    
    for (model, params, scale) in models:        
        grid_params = adjust_params(params)
        
        cv = get_gscv(model, grid_params, scale=scale)
        cv.fit(x_train, y_train)
        
        logging.info(model.__class__.__name__)
        logging.info("Best F1 Score")
        logging.info(cv.best_score_)
        logging.info("Best Parameters")
        logging.info(cv.best_params_)
        logging.info("Performance on Test Set")
        logging.info(cv.score(x_test, y_test))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
