# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 00:06:20 2018

@author: Deepak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
test_images = test.iloc[:, 0:].values.astype('float32')

train_images = dataset.iloc[:, 1:].values.astype('float32')
train_labels = dataset.iloc[:, 0:1].values.astype('int32')

sc = StandardScaler()
train_images = sc.fit_transform(train_images)
test_images = sc.fit_transform(test_images)

ohe = OneHotEncoder(categorical_features = [0])
train_labels = ohe.fit_transform(train_labels).toarray()

classifier = Sequential()
classifier.add(Dense(64, activation = 'relu', input_dim = (28 * 28)))
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dropout(0.15))
classifier.add(Dense(64, activation = 'relu'))
classifier.add(Dropout(0.15))
classifier.add(Dense(10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(train_images, train_labels, epochs = 100, batch_size = 64)


predict = classifier.predict_classes(test_images, verbose=0)
submissions = pd.DataFrame({'ImageId':list(range(1,len(predict) + 1)), "Label": predict})
submissions.to_csv("results.csv", index=False, header=True)