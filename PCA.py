import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd


def take_X_Y(WWC):
    Y = WWC['more_new_cases']
    X = WWC.drop(['Unnamed: 0.1.1', 'location', 'date', 'more_new_cases'], axis=1)
    return X, Y


if __name__ == '__main__':

    WWC = pd.read_csv('WWC_all_data_clean_09_01_2021_replace_nulls_with_average.csv')

    n_neighbors = 3
    random_state = 0

    # Load Digits dataset
    X, y = take_X_Y(WWC)

    # Split into train/test
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.5, stratify=y,
                         random_state=random_state)

    # dim = len(X.columns.size)
    n_classes = len(np.unique(y))

    # Reduce dimension to 2 with PCA
    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=2, random_state=random_state))

    # Reduce dimension to 2 with LinearDiscriminantAnalysis
    lda = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(n_components=2))

    # Reduce dimension to 2 with NeighborhoodComponentAnalysis
    nca = make_pipeline(StandardScaler(),
                        NeighborhoodComponentsAnalysis(n_components=2,
                                                       random_state=random_state))

    # Use a nearest neighbor classifier to evaluate the methods
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Make a list of the methods to be compared
    dim_reduction_methods = [('PCA', pca), ('LDA', lda), ('NCA', nca)]

    # plt.figure()
    for i, (name, model) in enumerate(dim_reduction_methods):
        plt.figure()
        # plt.subplot(1, 3, i + 1, aspect=1)

        # Fit the method's model
        model.fit(X_train, y_train)

        # Fit a nearest neighbor classifier on the embedded training set
        knn.fit(model.transform(X_train), y_train)

        # Compute the nearest neighbor accuracy on the embedded test set
        acc_knn = knn.score(model.transform(X_test), y_test)
        print(acc_knn)

        # Embed the data set in 2 dimensions using the fitted model
        X_embedded = model.transform(X)

        # Plot the projected points and show the evaluation score
        # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
        # plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name,
        #                                                           n_neighbors,
        #                                                           acc_knn))
    # plt.show()

    print("END")