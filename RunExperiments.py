from DT import DT_operation
from RandomForestTrees import RandomForestExpTrees
from ExtremelyRandomizedTree import ExtremlyRandomizedTree_Exp
from KNN import *
from NaiveBayes import *

if __name__ == '__main__':
    #-------------DecisionTree-------------
    # run the 4 decision tree tests (with different params) and prints the results
    # 1. Data Set: NoNulls, criterion: gini
    # 2. Data Set: AVG, criterion: gini
    # 3. Data Set: NoNulls, criterion: entropy
    # 4. Data Set: AVG, criterion: entropy
    DT_operation()

    # -------------RandomForest-------------
    # run the 4 RandomForest tests (with different params) and prints the results
    # Saves the results in <string param>.csv file
    # 1. Data Set: NoNulls, criterion: gini
    # 2. Data Set: AVG, criterion: gini
    # 3. Data Set: NoNulls, criterion: entropy
    # 4. Data Set: AVG, criterion: entropy
    # DT_operation()
    RandomForestExpTrees("RFT_Scores")

    # --------ExtremelyRandomizedTree-------
    # run the 4 RandomForest tests (with different params) and prints the results
    # Saves the results in <string param>.csv file
    # 1. Data Set: NoNulls, criterion: gini
    # 2. Data Set: AVG, criterion: gini
    # 3. Data Set: NoNulls, criterion: entropy
    # 4. Data Set: AVG, criterion: entropy
    # DT_operation()
    ExtremlyRandomizedTree_Exp("ERT_scores")

    # ------------------KNN-----------------
    # All functions save the results in <string param>.csv file
    # check knn on 1-100 neighbours in both data sets NoNulls and AVG
    KNNExp1("KNN_NoNulls_vs_AVG")

    # check knn on 1-100 neighbours in both normalized data sets NoNulls and AVG
    KNNExp2("KNN_NoNulls_vs_AVG_Normalized")

    # check knn on 1-100 neighbours in NoNulls dataSet classified (4, 8, 16, 32, 64, 128, 256)
    KNNExp3("KNN_NoNulls_Classified_first_classes")

    # check knn on 1-100 neighbours in AVG dataSet classified (4, 8, 16, 32, 64, 128, 256)
    KNNExp4("KNN_AVG_Classified_first_classes")

    # check knn on 1-100 neighbours in NoNulls dataSet classified (3000-10000 [jump of 1000])
    KNNExp5("KNN_NoNulls_Classified_second_classes")

    # check knn on 1-100 neighbours in NoNulls dataSet classified (3000-10000 [jump of 1000])
    KNNExp6("KNN_AVG_Classified_second_classes")

    # check knn on 1-100 neighbours in NoNulls dataSet classified (20000-100000 [jump of 20000])
    KNNExp7("KNN_NoNulls_Classified_third_classes")

    # check knn on 1-100 neighbours in AVG dataSet classified (20000-100000 [jump of 20000])
    KNNExp8("KNN_AVG_Classified_third_classes")

    # check knn on different number of neighbours in AVG 100000 classes
    KNNExp9("KNN_AVG_different_classes")

    # check knn on different number of neighbours in both AVG 100000 classes and AVG datasets in the k best features
    # chi-square function
    KNNExp10("KNN_KBest_chi_square")

    # check knn on different number of neighbours in both AVG 100000 classes and AVG datasets in the k best features
    # classif function
    KNNExp11("KNN_KBest_classif")

    # --------------NaiveBayes--------------
    # All functions save the results in <string param>.csv file
    # check NaiveBayes on different dataSets with multiNominal method
    NaiveBayesExp1("NaiveBayes_MultiNominal")

    # check NaiveBayes on different dataSets with bernouli method
    NaiveBayesExp2("NaiveBayes_Bernoli")

    # check NaiveBayes with multiNominal method on 100000 classes AVG dataset and different k best features
    NaiveBayesExp3("NaiveBayes_100000_classes_Multinominal_KBest")