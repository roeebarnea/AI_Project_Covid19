import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

    print(scores)
    print(sum(scores) / len(scores))
    return sum(scores) / len(scores)

def knn_graph(WWC, neis, dataSetName):
    final_scores = [dataSetName]
    for k in range(1, neis):
        final_scores.append(knn(WWC, k, 10))
    return final_scores


def pca_knn(WWC, neis, components):
    mat = []
    X, Y = take_X_Y(WWC)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    for c in components:
        row = [str(c)]
        pca = make_pipeline(StandardScaler(),
                            PCA(n_components=c, random_state=0))
        for n in neis:
            knn = KNeighborsClassifier(n_neighbors=n)
            pca.fit(X_train, y_train)
            knn.fit(pca.transform(X_train), y_train)
            acc_knn = knn.score(pca.transform(X_test), y_test)
            row.append(round(acc_knn*100, 2))
        mat.append(row)
    print(mat)
    return mat

def lda_knn(WWC, neis, components):
    mat = []
    X, Y = take_X_Y(WWC)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    for c in components:
        row = [str(c)]
        lda = make_pipeline(StandardScaler(),
                            LinearDiscriminantAnalysis(n_components=c))
        for n in neis:
            knn = KNeighborsClassifier(n_neighbors=n)
            lda.fit(X_train, y_train)
            knn.fit(lda.transform(X_train), y_train)
            acc_knn = knn.score(lda.transform(X_test), y_test)
            row.append(round(acc_knn*100, 2))
        mat.append(row)
    print(mat)
    return mat

def nca_knn(WWC, neis, components):
    mat = []
    X, Y = take_X_Y(WWC)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    for c in components:
        row = [str(c)]
        nca = make_pipeline(StandardScaler(),
                        NeighborhoodComponentsAnalysis(n_components=c,
                                                       random_state=0))

        for n in neis:
            knn = KNeighborsClassifier(n_neighbors=n)
            nca.fit(X_train, y_train)
            knn.fit(nca.transform(X_train), y_train)
            acc_knn = knn.score(nca.transform(X_test), y_test)
            row.append(round(acc_knn*100, 2))
        mat.append(row)
    print(mat)
    return mat




if __name__ == '__main__':

    neis = [3,4,5,9,15,30]
    random_state = 0
    components = [2,3,5,9,15,20,30,40]


    col = ['n_components']
    for n in neis:
        col.append(str(n))


    # df_scores = pd.read_csv('WWC_09_01_2021_KNN_SCORES_ALL.csv')

    # no_nulls_
    WWC_no_nulls = pd.read_csv('WWC_all_data_clean_09_01_2021_no_nulls.csv')

    # no_nulls_AVG
    WWC_no_nulls_AVG = pd.read_csv('WWC_all_data_clean_09_01_2021_no_nulls.csv')

    # PCA
    no_nulls_pca = pca_knn(WWC_no_nulls, neis, components)
    df = pd.DataFrame(no_nulls_pca, columns=col)
    df.to_csv('WWC_09_01_2021_no_nulls_PCA.csv')

    no_nulls_AVG_pca = pca_knn(WWC_no_nulls_AVG, neis, components)
    df = pd.DataFrame(no_nulls_AVG_pca, columns=col)
    df.to_csv('WWC_09_01_2021_no_nulls_AVG_PCA.csv')

    # LDA
    no_nulls_lda = lda_knn(WWC_no_nulls, neis, [2])
    df = pd.DataFrame(no_nulls_lda, columns=col)
    df.to_csv('WWC_09_01_2021_no_nulls_LDA.csv')

    no_nulls_AVG_lda = lda_knn(WWC_no_nulls_AVG, neis, [2])
    df = pd.DataFrame(no_nulls_AVG_lda, columns=col)
    df.to_csv('WWC_09_01_2021_no_nulls_AVG_LDA.csv')

    # NCA
    no_nulls_nca = nca_knn(WWC_no_nulls, neis, components)
    df = pd.DataFrame(no_nulls_nca, columns=col)
    df.to_csv('WWC_09_01_2021_no_nulls_NCA.csv')

    no_nulls_AVG_nca = nca_knn(WWC_no_nulls_AVG, neis, components)
    df = pd.DataFrame(no_nulls_AVG_nca, columns=col)
    df.to_csv('WWC_09_01_2021_no_nulls_AVG_NCA.csv')



    print("DONE")