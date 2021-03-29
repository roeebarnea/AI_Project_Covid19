import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def take_X_Y(WWC):
    Y = WWC['more_new_cases']
    X = WWC.drop(['Unnamed: 0.1.1', 'location', 'date', 'more_new_cases'], axis=1)
    return X, Y

def dt(WWC, crit, min_sample_split, max_feat):
    X, Y = take_X_Y(WWC)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    decision_tree = DecisionTreeClassifier(criterion=crit, min_samples_split=min_sample_split, max_features= max_feat)
    decision_tree.fit(X_train, y_train)
    Y_pred = decision_tree.predict(X_test)
    # acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
    acc_decision_tree = round(metrics.accuracy_score(Y_pred, y_test) * 100, 2)
    return acc_decision_tree

def dt_couple_of_tests(WWC, crit, min_sample_split, max_feat, tests):
    scores = []
    for i in range(tests):
        scores.append(dt(WWC, crit, min_sample_split, max_feat))

    # print(scores)
    # print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

if __name__ == '__main__':

    sample_splits = [2, 5, 10, 20]
    max_features = [1.0, 0.75, 0.5, 0.25, 'sqrt']
    TESTS = 10


    WWC_name = 'WWC_all_data_clean_09_01_2021_no_nulls.csv'
    WWC = pd.read_csv(WWC_name)
    print('gini')
    print(WWC_name)
    print('samples splits:')
    print(sample_splits)
    for mf in max_features:
        print('max features: ' + str(mf))
        scores_row = []
        for ss in sample_splits:
            scores_row.append(dt_couple_of_tests(WWC, 'gini', ss, mf, TESTS))
        print(scores_row)

    print()
    print()

    WWC_name = 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv'
    WWC = pd.read_csv(WWC_name)
    print('gini')
    print(WWC_name)
    print('samples splits:')
    print(sample_splits)
    for mf in max_features:
        print('max features: ' + str(mf))
        scores_row = []
        for ss in sample_splits:
            scores_row.append(dt_couple_of_tests(WWC, 'gini', ss, mf, TESTS))
        print(scores_row)

    print()
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_16_classes.csv'
    WWC = pd.read_csv(WWC_name)
    print('entropy')
    print(WWC_name)
    print('samples splits:')
    print(sample_splits)
    for mf in max_features:
        print('max features: ' + str(mf))
        scores_row = []
        for ss in sample_splits:
            scores_row.append(dt_couple_of_tests(WWC, 'gini', ss, mf, TESTS))
        print(scores_row)

    print()
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_AVG_16_classes.csv'
    WWC = pd.read_csv(WWC_name)
    print('entropy')
    print(WWC_name)
    print('samples splits:')
    print(sample_splits)
    for mf in max_features:
        print('max features: ' + str(mf))
        scores_row = []
        for ss in sample_splits:
            scores_row.append(dt_couple_of_tests(WWC, 'gini', ss, mf, TESTS))
        print(scores_row)

    print()
    print()





    print ('END')