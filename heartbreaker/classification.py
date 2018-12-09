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
from sklearn.svm import SVC

import data_loader
import util
import plotting

import xgboost

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
from sklearn.feature_selection import SelectFromModel


seed = 754927

def classification(model, x_train, y_train, seed=seed):   
    scoring = {
            'acc': make_scorer(accuracy_score),
            'prec': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
            }
    pipeline = Pipeline([
            ('transformer', StandardScaler()), 
            ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear', random_state=seed))), 
            ('estimator', model)])
        
    kf = KFold(n_splits=10, random_state=seed)
    # This is a dictionary of outputs
    cv_results = cross_validate(pipeline, x_train, y_train, cv=kf, scoring=scoring, return_train_score=False, return_estimator=True)
    
    accuracy = np.mean(cv_results['test_acc'])
    precision = np.mean(cv_results['test_prec'])
    recall = np.mean(cv_results['test_recall'])
    f1 = np.mean(cv_results['test_f1'])
    
    logging.info(model.__class__.__name__)
    logging.info("Accuracy: %0.4f, Precision: %0.4f, Recall: %0.4f, F1: %0.4f" % (accuracy, precision, recall, f1))

    return cv_results['estimator']

def classification2(model, x_train, y_train, x_test, y_test, seed=seed):   
    pipeline = Pipeline([
            ('transformer', StandardScaler()), 
            ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear', random_state=seed))), 
            ('estimator', model)])

    pipeline.fit(x_train, y_train)
    
    y_pred = pipeline.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logging.info(model.__class__.__name__)
    logging.info("Accuracy: %0.4f, Precision: %0.4f, Recall: %0.4f, F1: %0.4f" % (accuracy, precision, recall, f1))

    return pipeline

def main():
    data = util.impute_by_col(data_loader.load_all_data(), np.mean)
    rates = data.pop('heart_disease_mortality')
    rates = util.continuous_to_categorical(rates, percentile_cutoff=75)
    x = data
    y = rates
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=seed)
    
    models = []
    
    logistic = LogisticRegression(penalty='l1', C=0.1, max_iter=1000, solver='liblinear', class_weight='balanced')
    models.append(logistic)
    
    dt = DecisionTreeClassifier(criterion="gini", max_depth=None, max_features=None, min_samples_leaf=0.01, class_weight='balanced') #gini loss
    models.append(dt)
    
    rf = RandomForestClassifier(criterion="entropy", max_depth=None, max_features=None, min_samples_leaf=0.001, n_estimators=1000, class_weight='balanced')
    models.append(rf)
    
    svc_rbf = SVC(kernel='rbf', C=1, random_state=seed, class_weight='balanced')
    svc_linear = SVC(kernel='linear', C=0.01, random_state=seed, class_weight='balanced')
    svc_sigmoid = SVC(kernel='sigmoid', C=0.1, random_state=seed, class_weight='balanced')
    models.append(svc_rbf)
    models.append(svc_linear)
    models.append(svc_sigmoid)
    
    models = []
    xgb = xgboost.XGBClassifier(learning_rate=1e-2, max_depth=6, n_estimators=350, random_state=seed, class_weights='balanced')
    models.append(xgb)
    
    trained_models = []
    for model in models:
        trained_models.append(classification(model, x_train, y_train, seed))
        trained_models.append(classification2(model, x_train, y_train, x_test, y_test))
    
    # plotting.plot_shap_tree_summary(trained_models[-1][1], x_train, x_test)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
