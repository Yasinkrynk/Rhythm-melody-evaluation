# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:59:39 2015

@author: mariapanteli
"""
"""Selection of classifiers"""

import numpy

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm
from sklearn import metrics

def classifyKNN(X, Y, nn=1, kfold=5):
    model = KNeighborsClassifier(n_neighbors=nn, metric='euclidean')      
    Y_pred = cross_val_predict(model, X, Y, cv=kfold)
    accuracies = []
    for i in range(len(Y_pred)):
        accuracies.append(metrics.accuracy_score([Y[i]], [Y_pred[i]]))
    return numpy.asarray(accuracies)

def classifySVM(X, Y, kfold=5):
    model = svm.SVC(kernel='linear')
    Y_pred = cross_val_predict(model, X, Y, cv=kfold)
    accuracies = []
    for i in range(len(Y_pred)):
        accuracies.append(metrics.accuracy_score([Y[i]], [Y_pred[i]]))
    return numpy.asarray(accuracies)

def classifyLDA(X, Y, kfold=5):
    model = LDA(n_components=20)
    Y_pred = cross_val_predict(model, X, Y, cv=kfold)
    accuracies = []
    for i in range(len(Y_pred)):
        accuracies.append(metrics.accuracy_score([Y[i]], [Y_pred[i]]))
    return numpy.asarray(accuracies)

def classifyNB(X, Y, kfold=5):
    model = GaussianNB()
    Y_pred = cross_val_predict(model, X, Y, cv=kfold)
    accuracies = []
    for i in range(len(Y_pred)):
        accuracies.append(metrics.accuracy_score([Y[i]], [Y_pred[i]]))
    return numpy.asarray(accuracies)

