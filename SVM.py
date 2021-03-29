import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics

def take_X_Y(WWC):
    Y = WWC['more_new_cases']
    X = WWC.drop(['Unnamed: 0.1.1', 'location', 'date', 'more_new_cases'], axis=1)
    return X, Y

def svm_exp(WWC, k):
    X, Y = take_X_Y(WWC)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    svc = svm.SVC(kernel=k)
    svc.fit(X_train, y_train)
    Y_pred = svc.predict(X_test)
    acc_decision_tree = round(metrics.accuracy_score(Y_pred, y_test) * 100, 2)
    return acc_decision_tree


if __name__ == '__main__':
    kernels = ['linear', 'poly', 'rbf', 'sigmoid','precomputed']

    WWC = pd.read_csv('WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv')
    for k in kernels:
        print(k)
        print(svm_exp(WWC, k))




    print ('END')