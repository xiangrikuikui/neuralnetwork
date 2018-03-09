# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:56:16 2017

@author: lenovo
"""

import mnist_loader

from sklearn import svm

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    #train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    #test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))
    
if __name__ == "__main__":
    svm_baseline()