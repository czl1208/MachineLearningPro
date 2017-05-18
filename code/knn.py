#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:34:11 2017

@author: sunyue
"""

from sklearn.neighbors import KNeighborsClassifier
from numpy import array,load, count_nonzero
  
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
    
knn = KNeighborsClassifier()
knn.fit(train_data, train_labels) 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')

N = len(test_labels)
predict_data = knn.predict(test_data)
accuracy = (N-count_nonzero(predict_data-test_labels))/N

print(accuracy)