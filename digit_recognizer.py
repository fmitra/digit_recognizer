""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://www.kaggle.com/c/digit-recognizer

Competition goal is to take an image of a handwritten single digit, 
and determine what that digit is. The data for this competition were taken from 
the MNIST dataset.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import json
import os

# Possible algorithms we will evaluate for the prediction
ALGOS = [
        ('RandomForestClassifier', RandomForestClassifier),
        ('KNeighborsClassifier', KNeighborsClassifier),
        ('SVC', SVC),
        ('LinearSVC', LinearSVC),
        ('SGDClassifier', SGDClassifier),
        ('GaussianNB', GaussianNB)
    ]


def load_data():
    sample_train = pd.read_csv('data/train.csv', header=0)
    sample_test  = pd.read_csv('data/test.csv', header=0)

    return (sample_train, sample_test)


def test_algos(data_set, features):
    """
    Test the accuracy of the model by validating a a model against 
    the trained data set
    
    :data_set DataFrame: trained dataset with label column defined
    :features list: data_set features to evaluate
    :return list: Scores of evaluated algorithms and classifiers
    """
    if os.path.exists('data/scores.txt'):
        with open('data/scores.txt', 'r') as f:
            results = eval(f.read())
            return results

    # If a cache isn't found, separate teh data_set into
    # training and test sample and test the accuracy of each
    # algorithm
    train, test = train_test_split(data_set, test_size=0.3)

    results = []
    for name, algo in ALGOS:
        print('Running %s' % (name))
        model = algo()
        if 'n_jobs' in model.get_params():
            model = algo(n_jobs=-1)

        model.fit(train[features], train['label'])
        pred = model.predict(test[features])
        score = accuracy_score(test['label'].values,pred)
        print('%s accuracy at %s' % (name, str(score)))
        results.append((name, score))

    with open('data/scores.txt', 'w') as scores:
        scores.write(str(results))

    return results


def eval_scores(scores):
    """
    Checks the highest performing algorithm to use
    for the prediction

    :scores list: list of tuples containing evaluated algorithms and their scores
    :return tuple: model to work with 
    """
    # Find the highest scoring ML algorithm
    high_score = max(scores, key=lambda x: x[1])
    # Grab the corresponding model
    winner = [x for x in ALGOS if x[0] == high_score[0]][0]
    algo   = winner[0]
    model  = winner[1]

    return (algo, model)


def eval_kneighbors(train, test, features):
    """
    Start a grid search to find the optimal parameters for the 
    KNeighbors model
    
    :train DataFrame: trained dataset with label column defined
    :test DataFrame: test dataset with no label column
    :features list: data_set features to evaluate
    """
    metrics      = ['minkowski','euclidean'] 
    weights      = ['uniform','distance'] 
    neighbors    = np.arange(3,9)
    param_grid   = dict(metric=metrics,weights=weights,n_neighbors=neighbors,n_jobs=[-1])
    # Running the gridsearch with the full 48k list will take too long
    train_subset = train[0:15000]

    print("Fitting training data and evaluating optimal params for model...")
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=2, n_jobs=-1, verbose=10)
    grid.fit(train_subset[features], train_subset['label'])
    cv_results = grid.cv_results_

    print(cv_results)

    print("Storing param results...")
    with open('data/settings.txt', 'w') as settings:
        print(str(grid.best_estimator_))
        settings.write(str(grid.best_estimator_))


def train_kneighbors(train, test, features):
    """
    :train DataFrame: trained dataset with label column defined
    :test DataFrame: test dataset with no label column
    :features list: data_set features to evaluate
    """
    model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                 metric_params=None, n_jobs=-1, n_neighbors=4, p=2,
                                 weights='distance')

    print('Training model...')
    model.fit(train[features], train['label'])

    print('Running predictions...')
    pred = model.predict(test[features])
    results = pd.Series(pred)
    results.index += 1
    results.to_csv('data/results.csv')


def start():
    train, test = load_data()
    features = test.columns

    # Evaluate various ML algorithms with a subset of the data
    scores = test_algos(train, features)
    # Determine the model to use
    algo, model = eval_scores(scores)
    print('Running model for %s' % algo)

    # eval_kneighbors(train, test, features)
    train_kneighbors(train, test, features)

if __name__ == '__main__':
    start()

