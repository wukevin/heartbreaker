"""
Code for a logistic regression baseline 

When executed, 
"""

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

# Used below links as guidance for how to do multiple metric evaluation + K-fold CV
# https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
# https://stackoverflow.com/questions/46598301/how-to-compute-precision-recall-and-f1-score-of-an-imbalanced-dataset-for-k-fold

# and below links for guidance on how combine with scaling step in an sklearn pipeline
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#examples-using-sklearn-pipeline-pipeline
# https://stackoverflow.com/questions/44446501/how-to-standardize-data-with-sklearns-cross-val-score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.pipeline import Pipeline


seed = 754927

def classification(model, x_train, y_train, seed=seed, scale=False):   
    scoring = {
            'acc': make_scorer(accuracy_score),
            'prec': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
            }
    if (scale):
        pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', model)])
    else:
        pipeline = model
        
    kf = KFold(n_splits=10, random_state=seed)
    cv_results = cross_validate(pipeline, x_train, y_train, cv=kf, scoring=scoring, return_train_score=False)
    
    accuracy = np.mean(cv_results['test_acc'])
    precision = np.mean(cv_results['test_prec'])
    recall = np.mean(cv_results['test_recall'])
    f1 = np.mean(cv_results['test_f1'])
    
    logging.info(model.__class__.__name__)
    logging.info("Accuracy: %0.4f, Precision: %0.4f, Recall: %0.4f, F1: %0.4f" % (accuracy, precision, recall, f1))

def main():
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates = util.continuous_to_categorical(rates, percentile_cutoff=75)
    x = data
    y = rates
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=seed)
    
    models = []
    
    logistic = LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear')
    models.append((logistic, True))
    
    dt = DecisionTreeClassifier(min_samples_leaf=.01) #gini loss
    models.append((dt, False))
    
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=.01, bootstrap=True)
    models.append((rf, False))
    
    for (model, scale) in models:
        classification(model, x_train, y_train, seed, scale=scale)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
