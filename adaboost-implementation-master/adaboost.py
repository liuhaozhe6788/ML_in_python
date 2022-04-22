import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic
import math

""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_accuracy(pred, y):
    return sum(pred == y)/ float(len(y))

""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""
def print_accuracy(err):
    print("Accuracy: Training: %.4f - Test: %.4f" % err)

""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""
def generic_clf(y_train, X_train, y_test, X_test, clf):
    clf.fit(X_train, y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_accuracy(pred_train, y_train), get_accuracy(pred_test, y_test)

""" ADABOOST IMPLEMENTATION ================================================="""
def adaboost_clf(y_train, X_train, y_test, X_test, T, clf):
    n_train, n_test = len(X_train), len(X_test)

    # Initialize weights
    w = np.ones(n_train)/n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    accu_train, accu_test = [], []

    for i in range(T):
        # Fit a classifier with the specific weights
        clf.fit(X_train, y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)

        # Error
        err_m = sum(w[np.array(pred_train_i) != np.array(list(y_train))])
        # Alpha
        alpha_m = 0.5*np.log((1-err_m)/float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x)*alpha_m for x in -1*np.multiply(pred_train_i, list(y_train))])/(2*math.sqrt(err_m*(1-err_m))))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x*alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, [x*alpha_m for x in pred_test_i])]

        if i % 10 == 0:
            pred_train_adaboost_i, pred_test_adaboost_i = np.where(np.array(pred_train) > 0, 1, 0), np.where(np.array(pred_test) > 0, 1, 0)
            accu_train.append(get_accuracy(pred_train_adaboost_i, y_train))
            accu_test.append(get_accuracy(pred_test_adaboost_i, y_test))
    return accu_train, accu_test

""" PLOT FUNCTION ==========================================================="""
def plot_accu(accu_train, accu_test):
    df_accu = pd.DataFrame({
        "iter": list(range(len(accu_train))) * 2,
        "accu": accu_train + accu_test,
        "col": ["train"] * len(accu_train) + ["test"] * len(accu_test)
    })
    ax = sns.lineplot(x="iter", y="accu", hue="col", data=df_accu, marker="o")
    plt.show()

""" MAIN SCRIPT ============================================================="""
if __name__ == "__main__":

    # read data
    data = load_breast_cancer()
    df = pd.DataFrame(data=np.c_[data.data, data.target], columns=np.append(data['feature_names'], ['target']))
    malignant_df = df[df["target"] == 0].sample(frac=1)
    benign_df = df[df["target"] == 1]
    df = pd.concat([malignant_df, benign_df])
    print(len(df[df["target"] == 0]), len(df[df["target"] == 1]))

    # split into training and test set
    train, test = train_test_split(df, test_size=0.3)
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth=math.pow(2, 61), random_state=1)
    er_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    clf_res = generic_clf(y_train, X_train, y_test, X_test, clf_tree)
    # print_accuracy(clf_res)

    # Fit AdaBoost clf using a decision tree as a base estimator
    accu_train, accu_test = adaboost_clf(y_train, X_train, y_test, X_test, 400, er_tree)
    plot_accu(accu_train, accu_test)
