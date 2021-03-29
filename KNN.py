import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn import metrics

def take_X_Y(WWC):
    Y = WWC['more_new_cases']
    X = WWC.drop(['Unnamed: 0.1.1', 'location', 'date', 'more_new_cases'], axis=1)
    return X, Y

def knn(WWC, k, tests):
    scores = []
    for i in range(tests):
        X, Y = take_X_Y(WWC)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        Y_pred = classifier.predict(X_test)
        #print(confusion_matrix(y_test, y_pred))
        #print(classification_report(y_test, y_pred))

        acc_knn = round(metrics.accuracy_score(y_test, Y_pred) * 100, 2)
        scores.append(acc_knn)
        #print(acc_knn)

    # print(scores)
    # print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

def knn_selectKBest(WWC, neis, k, tests):
    scores = []
    for i in range(tests):
        X, Y = take_X_Y(WWC)
        X_new = SelectKBest(f_classif, k=k).fit_transform(X, Y)
        X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.20)
        classifier = KNeighborsClassifier(n_neighbors=neis)
        classifier.fit(X_train, y_train)
        Y_pred = classifier.predict(X_test)
        #print(confusion_matrix(y_test, y_pred))
        #print(classification_report(y_test, y_pred))

        acc_knn = round(metrics.accuracy_score(y_test, Y_pred) * 100, 2)
        scores.append(acc_knn)
        #print(acc_knn)

    print(scores)
    print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

def knn_selectKBest_chi2(WWC, neis, k, tests):
    scores = []
    for i in range(tests):
        X, Y = take_X_Y(WWC)
        X_new = SelectKBest(chi2, k=k).fit_transform(X, Y)
        X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.20)
        classifier = KNeighborsClassifier(n_neighbors=neis)
        classifier.fit(X_train, y_train)
        Y_pred = classifier.predict(X_test)
        #print(confusion_matrix(y_test, y_pred))
        #print(classification_report(y_test, y_pred))

        acc_knn = round(metrics.accuracy_score(y_test, Y_pred) * 100, 2)
        scores.append(acc_knn)
        #print(acc_knn)

    print(scores)
    print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

def knn_graph(WWC, neis, dataSetName):
    final_scores = [dataSetName]
    for k in range(1, neis):
        final_scores.append(knn(WWC, k, 3))

    print(final_scores)
    return final_scores

def knn_Exp(WWC, neis, dataSetName, tests):
    print("knn exp. file name: " + dataSetName)
    final_scores = [dataSetName]
    for k in range(1, neis):
        final_scores.append(knn(WWC, k, tests))

    print(final_scores)
    return final_scores

def knn_graph_classes(data_name, classes, neis, mat):
    scores = []
    for c in classes:
        dataSetName = data_name + '_' + str(c) + '_classes.csv'
        WWC_no_nulls = pd.read_csv(dataSetName)
        scores = knn_graph(WWC_no_nulls, neis, dataSetName)
        mat.append(scores)

def knn_classes_Exp(data_name, classes, neis, mat, tests):
    scores = []
    for c in classes:
        dataSetName = data_name + '_' + str(c) + '_classes.csv'
        WWC_no_nulls = pd.read_csv(dataSetName)
        scores = knn_Exp(WWC_no_nulls, neis, dataSetName, tests)
        mat.append(scores)

def NormalizedExp(mat):
    WWC_no_nulls_Nor = pd.read_csv('WWC_09_01_2021_no_nulls_NORMALIZE.csv')
    WWC_no_nulls_AVG_Nor = pd.read_csv('WWC_09_01_2021_no_nulls_AVG_NORMALIZE.csv')

    mat.append(knn_graph(WWC_no_nulls_Nor, neis, 'WWC_09_01_2021_no_nulls_NORMALIZE.csv'))
    mat.append(knn_graph(WWC_no_nulls_AVG_Nor, neis, 'WWC_09_01_2021_no_nulls_AVG_NORMALIZE.csv'))

def knn_graph_classes_2(data_name, classes, neis, mat):
    for c in classes:
        dataSetName = data_name + '_' + str(c) + '_classes.csv'
        WWC_no_nulls = pd.read_csv(dataSetName)
        final_scores = [dataSetName]
        for n in neis:
            final_scores.append(knn(WWC_no_nulls, n, 3))
        mat.append(final_scores)

def analize_knn_8000_neis():
    neis = [1, 2, 5, 20, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2000, 2250, 2500, 2750, 3000, 3250, 3500,
            3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000]

    col = ['dataSetName']
    for i in neis:
        col.append(str(i) + '_k')

    classes = [100000]

    mat = []
    knn_graph_classes_2("WWC_09_01_2021_no_nulls_AVG", classes, neis, mat)

    print(mat)
    df = pd.DataFrame(mat, columns=col)
    df.to_csv('WWC_09_01_2021_KNN_Scores_16000_neis_3csv')

def invesitgate_select_k_Best_f_classif():
    neis = [1, 2, 5, 20, 50, 100, 250, 500]
    k_best = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90 ]
    col = ['dataSetName + neis']
    for i in k_best:
        col.append(str(i))

    DataSetsNames = ['WWC_09_01_2021_no_nulls_AVG_100000_classes.csv',
                     'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv']

    mat = []

    for ds in DataSetsNames:
        df = pd.read_csv(ds)
        for n in neis:
            row = [(ds + " " + str(n))]
            for kb in k_best:
                row.append(knn_selectKBest(df, n, kb, 3))
            mat.append(row)

    print(mat)
    df = pd.DataFrame(mat, columns=col)
    df.to_csv('WWC_09_01_2021_KNN_k_best_neis_' + 'f_clasif' + '.csv')

def invesitgate_select_k_Best_chi2():
    neis = [1, 2, 5, 20, 50, 100, 250, 500]
    k_best = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90 ]
    col = ['dataSetName + neis']
    for i in k_best:
        col.append(str(i))

    DataSetsNames = ['WWC_09_01_2021_no_nulls_AVG_100000_classes.csv',
                     'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv']

    mat = []

    for ds in DataSetsNames:
        df = pd.read_csv(ds)
        for n in neis:
            row = [(ds + " " + str(n))]
            for kb in k_best:
                row.append(knn_selectKBest_chi2(df, n, kb, 3))
            mat.append(row)

    print(mat)
    df = pd.DataFrame(mat, columns=col)
    df.to_csv('WWC_09_01_2021_KNN_k_best_neis_' + 'chi_2' + '.csv')

def KNNExp1(file_name):
    mat = []
    neis = 101
    tests = 10
    col = ['dataSetName']
    for i in range(1, neis):
        col.append(str(i) + '_k')
    WWC_no_nulls = pd.read_csv('WWC_all_data_clean_09_01_2021_no_nulls.csv')
    WWC_no_nulls_AVG = pd.read_csv('WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv')

    mat.append(knn_Exp(WWC_no_nulls, neis, 'WWC_all_data_clean_09_01_2021_no_nulls.csv', tests))
    mat.append(knn_Exp(WWC_no_nulls_AVG, neis, 'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv', tests))

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp2(file_name):
    mat = []
    neis = 101
    tests = 10
    col = ['dataSetName']
    for i in range(1, neis):
        col.append(str(i) + '_k')
    WWC_no_nulls = pd.read_csv('WWC_09_01_2021_no_nulls_NORMALIZE.csv')
    WWC_no_nulls_AVG = pd.read_csv('WWC_09_01_2021_no_nulls_AVG_NORMALIZE.csv')

    mat.append(knn_Exp(WWC_no_nulls, neis, 'WWC_09_01_2021_no_nulls_NORMALIZE.csv', tests))
    mat.append(knn_Exp(WWC_no_nulls_AVG, neis, 'WWC_09_01_2021_no_nulls_AVG_NORMALIZE.csv', tests))

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp3(file_name):
    mat = []
    neis = 101
    tests = 10
    classes = [4, 8, 16, 32, 64, 128, 256]
    col = ['dataSetName']
    for i in range(1, neis):
        col.append(str(i) + '_k')
    knn_classes_Exp('WWC_09_01_2021_no_nulls', classes, neis, mat, tests)

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp4(file_name):
    mat = []
    neis = 101
    tests = 10
    classes = [4, 8, 16, 32, 64, 128, 256]
    col = ['dataSetName']
    for i in range(1, neis):
        col.append(str(i) + '_k')
    knn_classes_Exp('WWC_09_01_2021_no_nulls_AVG', classes, neis, mat, tests)

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp5(file_name):
    mat = []
    neis = 101
    tests = 10
    classes = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    col = ['dataSetName']
    for i in range(1, neis):
        col.append(str(i) + '_k')
    knn_classes_Exp('WWC_09_01_2021_no_nulls', classes, neis, mat, tests)

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp6(file_name):
    mat = []
    neis = 101
    tests = 10
    classes = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    col = ['dataSetName']
    for i in range(1, neis):
        col.append(str(i) + '_k')
    knn_classes_Exp('WWC_09_01_2021_no_nulls_AVG', classes, neis, mat, tests)

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp7(file_name):
    mat = []
    neis = 101
    tests = 10
    classes = [20000, 40000, 60000, 80000, 100000]
    col = ['dataSetName']
    for i in range(1, neis):
        col.append(str(i) + '_k')
    knn_classes_Exp('WWC_09_01_2021_no_nulls', classes, neis, mat, tests)

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp8(file_name):
    mat = []
    neis = 101
    tests = 10
    classes = [20000, 40000, 60000, 80000, 100000]
    col = ['dataSetName']
    for i in range(1, neis):
        col.append(str(i) + '_k')
    knn_classes_Exp('WWC_09_01_2021_no_nulls_AVG', classes, neis, mat, tests)

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp9(file_name):
    mat = []
    neis = [1, 2, 5, 20, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2000, 2250, 2500, 2750, 3000, 3250, 3500,
            3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000]
    tests = 10
    col = ['dataSetName']
    for i in neis:
        col.append(str(i) + '_k')

    WWC_AVG_100000 = pd.read_csv('WWC_09_01_2021_no_nulls_AVG_100000_classes.csv')
    row = ['WWC_09_01_2021_no_nulls_AVG_100000_classes.csv']
    for i in neis:
        row.append(knn(WWC_AVG_100000, i, tests))
    print(row)
    mat.append(row)

    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp10(file_name):
    tests = 10
    neis = [1, 2, 5, 20, 50, 100, 250, 500]
    k_best = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    col = ['dataSetName + neis']
    for i in k_best:
        col.append(str(i))

    DataSetsNames = ['WWC_09_01_2021_no_nulls_AVG_100000_classes.csv',
                     'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv']

    mat = []

    for ds in DataSetsNames:
        df = pd.read_csv(ds)
        for n in neis:
            row = [(ds + " " + str(n))]
            for kb in k_best:
                row.append(knn_selectKBest_chi2(df, n, kb, tests))
            print(row)
            mat.append(row)

    print(mat)
    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

def KNNExp11(file_name):
    tests = 10
    neis = [1, 2, 5, 20, 50, 100, 250, 500]
    k_best = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]
    col = ['dataSetName + neis']
    for i in k_best:
        col.append(str(i))

    DataSetsNames = ['WWC_09_01_2021_no_nulls_AVG_100000_classes.csv',
                     'WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv']

    mat = []

    for ds in DataSetsNames:
        df = pd.read_csv(ds)
        for n in neis:
            row = [(ds + " " + str(n))]
            for kb in k_best:
                row.append(knn_selectKBest(df, n, kb, tests))
            mat.append(row)

    print(mat)
    df = pd.DataFrame(mat, columns=col)
    df.to_csv(file_name + '.csv')

if __name__ == '__main__':

    print("DONE")