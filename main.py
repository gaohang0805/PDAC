# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
from sklearn.linear_model import Lasso
import copy
import operator
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

import random
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def readData(file_name):
    rawData = pd.read_csv(file_name)
    return rawData


def readLabel(file_name):
    labelData = pd.read_csv(file_name, encoding="gbk", usecols=[10])
    return labelData


def readtestLabel(file_name):
    labelData = pd.read_csv(file_name, encoding="gbk", usecols=[6])
    return labelData


def main():

    # read data
    rawData = readData("data.csv")
    rawData.pop("encode")
    rawData.pop("Image")
    rawData.pop("ROI ")
    rawData.pop("MRN ")
    where_are_nan = np.isnan(rawData)
    where_are_inf = np.isinf(rawData)
    head_name = rawData.columns[2:]
    t_data = readData("1.csv")
    t_data.pop('Index')
    t_data.pop("Image")
    t_data.pop("ROI ")
    t_data.pop("MRN ")

    t_where_are_nan = np.isnan(t_data)
    t_where_are_inf = np.isinf(t_data)
    t_head_name = t_data.columns[2:]

    data = np.asarray(rawData)
    tn_data = np.asarray(t_data)

    where_are_nan = np.isnan(rawData)
    where_are_inf = np.isinf(rawData)
    data[where_are_inf] = 0
    data[where_are_nan] = 0
    tn_data[t_where_are_inf] = 0
    tn_data[t_where_are_nan] = 0
    data = list(data)
    tn_data = list(tn_data)
    #data = list(data) + tn_data

    # read label
    labelData = readLabel("label.csv")
    t_labelData = readtestLabel("label1.csv")
    label = list(np.asarray(labelData))
    t_label = list(np.asarray(t_labelData))

    # dif_len = len(data) - len(label)
    # for i in range(dif_len):
    #     label.append('低')

    x, y = [], []
    t = 0
    for i, j in zip(data, label):

        if type(j[0]) == str:
            if(j[0] == '中' or j[0] == '高' or j == '中' or j == '高'):
                y.append('高')
                x.append(list(i))
            elif j[0] == '低' or j == '低':
                y.append(list(j)[0])
                x.append(list(i))
        else:
            t += 1

    y = LabelEncoder().fit_transform(y)

    x = StandardScaler().fit_transform(x)
    sel = VarianceThreshold(threshold=0.5)
    x = sel.fit_transform(x)
    cc = 1.2

    model = SVC(kernel="linear", C=cc)#预测模型
    rfe = RFE(estimator=model, n_features_to_select=36, step=500)
    #用于特征选择

    x_train, x_test = [], []
    y_train, y_test = [], []
    t = 0

    for i, j in zip(x, y):
        if t % 2 == 0:
            x_train.append(i)
            y_train.append(j)
        else:
            x_test.append(i)
            y_test.append(j)
        t += 1

    rfe.fit(x, y)

    x_train = rfe.transform(x_train)
    x_test = rfe.transform(x_test)
    x = rfe.transform(x)

    clf = SVC(kernel="linear", C=cc)

    clf.fit(x, y)
    ans = clf.predict(x_train)
    y_pred = []

    for i in ans:
        if i > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred)
    print("AUC:%.4f" % metrics.auc(fpr, tpr))
    print('ACC: %.4f' % metrics.accuracy_score(y_train, y_pred))

    print(len([i for i, j in zip(y_train, y_pred) if i
               == j and j == 0]) / len([i for i in y_train if i == 0]))
    print(len([i for i, j in zip(y_train, y_pred) if i
               == j and j == 1]) / len([i for i in y_train if i == 1]))
    ans = clf.predict(x_test)
    y_pred = []
    for i in ans:
        if i > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)


#    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    print("AUC:%.4f" % metrics.auc(fpr, tpr))
    print('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))

    print(len([i for i, j in zip(y_test, y_pred) if i
               == j and j == 0]) / len([i for i in y_test if i == 0]))
    print(len([i for i, j in zip(y_test, y_pred) if i
               == j and j == 1]) / len([i for i in y_test if i == 1]))

#    print(y_test)


if __name__ == "__main__":
    main()
