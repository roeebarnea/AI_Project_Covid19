import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.naive_bayes import CategoricalNB

def take_X_Y(WWC):
    Y = WWC['more_new_cases']
    X = WWC.drop(['Unnamed: 0.1.1', 'location', 'date', 'more_new_cases'], axis=1)
    return X, Y

def NBmulti(WWC, a):
    X, Y = take_X_Y(WWC)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    NBmulti = MultinomialNB(alpha=a)
    NBmulti.fit(X_train, y_train)
    Y_pred = NBmulti.predict(X_test)
    # acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
    acc = round(metrics.accuracy_score(Y_pred, y_test) * 100, 2)
    return acc

def NBBernouli(WWC, a):
    X, Y = take_X_Y(WWC)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    NB = BernoulliNB(alpha=a, binarize= True)
    NB.fit(X_train, y_train)
    Y_pred = NB.predict(X_test)
    # acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
    acc = round(metrics.accuracy_score(Y_pred, y_test) * 100, 2)
    return acc

def NBmulti_couple_of_tests(WWC, alpha, tests):
    scores = []
    for i in range(tests):
        scores.append(NBmulti(WWC, alpha))

    print(scores)
    #print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

def NBBernouli_couple_of_tests(WWC, alpha, tests):
    scores = []
    for i in range(tests):
        scores.append(NBBernouli(WWC, alpha))

    print(scores)
    #print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

def NBmulti_selectKBest(WWC, a, k):
    X, Y = take_X_Y(WWC)
    X_new = SelectKBest(f_classif, k=k).fit_transform(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.20)
    NBmulti = MultinomialNB(alpha=a)
    NBmulti.fit(X_train, y_train)
    Y_pred = NBmulti.predict(X_test)
    # acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
    acc = round(metrics.accuracy_score(Y_pred, y_test) * 100, 2)
    return acc

def NBBernouli_selectKBest(WWC, a, k):
    X, Y = take_X_Y(WWC)
    X_new = SelectKBest(f_classif, k=k).fit_transform(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.20)
    NB = BernoulliNB(alpha=a, binarize= True)
    NB.fit(X_train, y_train)
    Y_pred = NB.predict(X_test)
    # acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
    acc = round(metrics.accuracy_score(Y_pred, y_test) * 100, 2)
    return acc

def NBmulti_couple_of_tests_selectKBest(WWC, alpha, tests, k):
    scores = []
    for i in range(tests):
        scores.append(NBmulti_selectKBest(WWC, alpha, k))

    print(scores)
    #print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

def NBBernouli_couple_of_tests_selectKBest(WWC, alpha, tests, k):
    scores = []
    for i in range(tests):
        scores.append(NBBernouli_selectKBest(WWC, alpha, k))

    print(scores)
    #print(sum(scores) / len(scores))
    return sum(scores) / len(scores)


def NBmulti_AVG_different_alphas():
    alphas = [1, 0.75, 0.5, 0.25, 0.01, 0.05, 0.001, 0.0001, 0.00001]
    col = ['dataSetName + alg']
    for i in alphas:
        col.append(str(i))
    TESTS = 10
    mat = []

    WWC_name = 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv'
    WWC = pd.read_csv(WWC_name)
    print(WWC_name)
    print(alphas)
    scores_row = ['AVG_NBMulti']
    for a in alphas:
        scores_row.append(NBmulti_couple_of_tests(WWC, a, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv'
    WWC = pd.read_csv(WWC_name)
    print(WWC_name)
    print(alphas)
    scores_row = ['AVG_NBBernouli']
    for a in alphas:
        scores_row.append(NBBernouli_couple_of_tests(WWC, a, TESTS))
    mat.append(scores_row)

    df = pd.DataFrame(mat, columns=col)
    df.to_csv('WWC_09_01_2021_NB_AVG.csv')

def NBmulti_AVG_different_alphas_classes():
    alphas = [1, 0.75, 0.5, 0.25, 0.01, 0.05, 0.001, 0.0001, 0.00001]
    col = ['dataSetName + alg']
    for i in alphas:
        col.append(str(i))
    TESTS = 10

    classes = [4, 16, 256, 512, 1000, 5000, 10000, 30000, 60000, 100000]

    mat = []

    for c in classes:
        WWC_name = 'WWC_09_01_2021_no_nulls_AVG_' + str(c) + '_classes.csv'
        WWC = pd.read_csv(WWC_name)

        scores_row = ['NBMulti_' + WWC_name]
        for a in alphas:
            scores_row.append(NBmulti_couple_of_tests(WWC, a, TESTS))
        mat.append(scores_row)
        print()

        scores_row = ['NBBernouli_' + WWC_name]
        for a in alphas:
            scores_row.append(NBBernouli_couple_of_tests(WWC, a, TESTS))
        mat.append(scores_row)

    df = pd.DataFrame(mat, columns=col)
    df.to_csv('WWC_09_01_2021_NB_classes.csv')

#1
def AVG_NB_Multi():
    classes = [4,16,256, 512, 1000, 5000, 8000, 10000, 40000, 80000, 100000]
    col = ['dataSetName + alg', 'success rate']
    TESTS = 10
    mat = []

    WWC_name = 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['AVG_NBMulti']
    scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_all_data_clean_09_01_2021_no_nulls.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['NoNulls_NBMulti']
    scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_NORMALIZE.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['NoNulls_Normalized']
    scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_AVG_NORMALIZE.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['AVG_Normalized']
    scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    for c in classes:
        WWC_name = 'WWC_09_01_2021_no_nulls_AVG_' + str(c) + '_classes.csv'
        WWC = pd.read_csv(WWC_name)
        scores_row = [WWC_name]
        scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
        mat.append(scores_row)
        print()

    df = pd.DataFrame(mat, columns=col)
    df.to_csv('WWC_09_01_2021_NB_MULTI.csv')

#2
def AVG_NB_Bernouli():
    classes = [4,16,256, 512, 1000, 5000, 8000, 10000, 40000, 80000, 100000]
    col = ['dataSetName + alg', 'success rate']
    TESTS = 10
    mat = []

    WWC_name = 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['AVG_NBMulti']
    scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_all_data_clean_09_01_2021_no_nulls.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['NoNulls_NBMulti']
    scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_NORMALIZE.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['NoNulls_Normalized']
    scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_AVG_NORMALIZE.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['AVG_Normalized']
    scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    for c in classes:
        WWC_name = 'WWC_09_01_2021_no_nulls_AVG_' + str(c) + '_classes.csv'
        WWC = pd.read_csv(WWC_name)
        scores_row = [WWC_name]
        scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
        mat.append(scores_row)
        print()

    df = pd.DataFrame(mat, columns=col)
    df.to_csv('WWC_09_01_2021_NB_Bernouli.csv')

#3
def AVG_100000_SelectK_NBmulti():
    k_best = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    col = ['k_best', 'success rate']
    TESTS = 10
    mat = []

    WWC_name = 'WWC_09_01_2021_no_nulls_AVG_100000_classes.csv'
    WWC = pd.read_csv(WWC_name)

    for k in k_best:
        scores_row = [str(k)]
        scores_row.append(NBmulti_couple_of_tests_selectKBest(WWC, 1.0, TESTS, k))
        mat.append(scores_row)
        print()

    df = pd.DataFrame(mat, columns=col)
    df.to_csv('WWC_09_01_2021_NB_MULTI_selectK.csv')

def NaiveBayesExp1(file_name):
    classes = [4, 16, 256, 512, 1000, 5000, 8000, 10000, 40000, 80000, 100000]
    col = ['dataSetName + alg', 'success rate']
    TESTS = 10
    mat = []

    WWC_name = 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['AVG_NBMulti']
    scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_all_data_clean_09_01_2021_no_nulls.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['NoNulls_NBMulti']
    scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_NORMALIZE.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['NoNulls_Normalized']
    scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_AVG_NORMALIZE.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['AVG_Normalized']
    scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    for c in classes:
        WWC_name = 'WWC_09_01_2021_no_nulls_AVG_' + str(c) + '_classes.csv'
        WWC = pd.read_csv(WWC_name)
        scores_row = [WWC_name]
        scores_row.append(NBmulti_couple_of_tests(WWC, 1.0, TESTS))
        mat.append(scores_row)
        print()

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def NaiveBayesExp2(file_name):
    classes = [4, 16, 256, 512, 1000, 5000, 8000, 10000, 40000, 80000, 100000]
    col = ['dataSetName + alg', 'success rate']
    TESTS = 10
    mat = []

    WWC_name = 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['AVG_NBMulti']
    scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_all_data_clean_09_01_2021_no_nulls.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['NoNulls_NBMulti']
    scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_NORMALIZE.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['NoNulls_Normalized']
    scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    WWC_name = 'WWC_09_01_2021_no_nulls_AVG_NORMALIZE.csv'
    WWC = pd.read_csv(WWC_name)
    scores_row = ['AVG_Normalized']
    scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
    mat.append(scores_row)
    print()

    for c in classes:
        WWC_name = 'WWC_09_01_2021_no_nulls_AVG_' + str(c) + '_classes.csv'
        WWC = pd.read_csv(WWC_name)
        scores_row = [WWC_name]
        scores_row.append(NBBernouli_couple_of_tests(WWC, 1.0, TESTS))
        mat.append(scores_row)
        print()

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def NaiveBayesExp3(file_name):
    k_best = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    col = ['k_best', 'success rate']
    TESTS = 10
    mat = []

    WWC_name = 'WWC_09_01_2021_no_nulls_AVG_100000_classes.csv'
    WWC = pd.read_csv(WWC_name)

    for k in k_best:
        scores_row = [str(k)]
        scores_row.append(NBmulti_couple_of_tests_selectKBest(WWC, 1.0, TESTS, k))
        mat.append(scores_row)
        print()

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')


if __name__ == '__main__':

    print('END')