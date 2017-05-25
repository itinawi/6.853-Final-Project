from sklearn import svm
import numpy as np


def svm_learn(x_values, y_values):
    clf = svm.SVC()
    clf.fit(x_values[:2000], y_values[:2000])
    prediction = clf.predict(x_values[2000:])
    actual = y_values[2000:]
    return sum(prediction==actual)/len(actual)