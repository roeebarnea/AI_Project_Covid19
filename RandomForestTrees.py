import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def take_X_Y(WWC):
    Y = WWC['more_new_cases']
    X = WWC.drop(['Unnamed: 0.1.1', 'location', 'date', 'more_new_cases'], axis=1)
    return X, Y

def random_forest(WWC, crit, min_sample_split, estim):
    X, Y = take_X_Y(WWC)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    decision_tree = RandomForestClassifier(criterion=crit, min_samples_split=min_sample_split, max_features=1.0, n_estimators= estim)
    decision_tree.fit(X_train, y_train)
    Y_pred = decision_tree.predict(X_test)
    # acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
    acc = round(metrics.accuracy_score(Y_pred, y_test) * 100, 2)
    return acc

def random_forest_couple_of_tests(WWC, crit, min_sample_split, estim, tests):
    scores = []
    for i in range(tests):
        scores.append(random_forest(WWC, crit, min_sample_split, estim))

    # print(scores)
    # print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

def RandomForestExpTrees(file_name):
    sample_splits = [2, 5, 10, 20, 30, 40]
    n_estimators = [1, 5, 10, 20, 30, 40, 50, 70, 90, 120]
    TESTS = 10
    mat = []

    WWC_name = 'WWC_all_data_clean_09_01_2021_no_nulls.csv'
    WWC = pd.read_csv(WWC_name)
    print('gini')
    print(WWC_name)
    print('samples splits:')
    print(sample_splits)
    for n in n_estimators:
        print('n_estimators: ' + str(n))
        scores_row = []
        for ss in sample_splits:
            scores_row.append(random_forest_couple_of_tests(WWC, 'gini', ss, n, TESTS))
        print(scores_row)
        mat.append(scores_row)
    print()
    print()

    WWC_name = 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv'
    WWC = pd.read_csv(WWC_name)
    print('gini')
    print(WWC_name)
    print('samples splits:')
    print(sample_splits)
    for n in n_estimators:
        print('n_estimators: ' + str(n))
        scores_row = []
        for ss in sample_splits:
            scores_row.append(random_forest_couple_of_tests(WWC, 'gini', ss, n, TESTS))
        print(scores_row)
        mat.append(scores_row)
    print()
    print()

    WWC_name = 'WWC_all_data_clean_09_01_2021_no_nulls.csv'
    WWC = pd.read_csv(WWC_name)
    print('entropy')
    print(WWC_name)
    print('samples splits:')
    print(sample_splits)
    for n in n_estimators:
        print('n_estimators: ' + str(n))
        scores_row = []
        for ss in sample_splits:
            scores_row.append(random_forest_couple_of_tests(WWC, 'entropy', ss, n, TESTS))
        print(scores_row)
        mat.append(scores_row)
    print()
    print()

    WWC_name = 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv'
    WWC = pd.read_csv(WWC_name)
    print('entropy')
    print(WWC_name)
    print('samples splits:')
    print(sample_splits)
    for n in n_estimators:
        print('n_estimators: ' + str(n))
        scores_row = []
        for ss in sample_splits:
            scores_row.append(random_forest_couple_of_tests(WWC, 'entropy', ss, n, TESTS))
        print(scores_row)
        mat.append(scores_row)
    print()
    print()

    df = pd.DataFrame(mat, columns=sample_splits)
    df.to_csv(file_name + '.csv')

if __name__ == '__main__':


    print ('END')