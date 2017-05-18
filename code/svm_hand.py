#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 09:49:26 2017

@author: sunyue
"""

from svmutil import *
from numpy import array,load,savetxt


    
train_data = list(load('train_data.npy'))
train_labels = list(load('train_label.npy'))
test_data = list(load('test_data.npy'))
test_labels = list(load('test_label.npy'))

for i in range(len(train_data)):
    train_data[i] = list(train_data[i])
    train_labels[i] = int(list(train_labels[i])[0])
for j in range(len(test_data)):
    test_data[j] = list(test_data[j])
    test_labels[j] = int(list(test_labels[j])[0])

prob = svm_problem(train_labels,train_data)
param = svm_parameter()
param.svm_type = 0
param.kernel_type = 1
#'''    -s svm_type : set type of SVM (default 0)
#        0 -- C-SVC      (multi-class classification)
#        1 -- nu-SVC     (multi-class classification)
#        2 -- one-class SVM
#        3 -- epsilon-SVR    (regression)
#        4 -- nu-SVR     (regression)
#    -t kernel_type : set type of kernel function (default 2)
#        0 -- linear: u'*v
#        1 -- polynomial: (gamma*u'*v + coef0)^degree
#        2 -- radial basis function: exp(-gamma*|u-v|^2)
#        3 -- sigmoid: tanh(gamma*u'*v + coef0)
#        4 -- precomputed kernel (kernel values in training_set_file)
#    -d degree : set degree in kernel function (default 3)
#    -g gamma : set gamma in kernel function (default 1/num_features)
#    -r coef0 : set coef0 in kernel function (default 0)
#    -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
#    -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
#    -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
#    -m cachesize : set cache memory size in MB (default 100)
#    -e epsilon : set tolerance of termination criterion (default 0.001)
#    -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
#    -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
#    -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
#    -v n: n-fold cross validation mode
#    -q : quiet mode (no outputs)'''
param.C = 100
param.degree = 2
param.gamma = 1
param.coef0 = 1
m = svm_train(prob, param)
p = svm_predict(test_labels,test_data,m)