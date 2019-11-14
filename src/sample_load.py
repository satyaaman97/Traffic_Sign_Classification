# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:42:08 2019

@author: Joon Kim
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# flag for grayscale
isGray = False
file = 'ToySet1_rgb.pickle'

# Opening 'pickle' file and getting images
with open(file, 'rb') as f:
    d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
    # At the same time method 'astype()' used for converting ndarray from int to float
    # It is needed to divide float by float when applying Normalization
    x_train = d['x_train'].astype(np.float32)
    y_train = d['y_train']
    x_validation = d['x_validation'].astype(np.float32)
    y_validation = d['y_validation']
    x_test = d['x_test'].astype(np.float32)
    y_test = d['y_test']

print('Sample Images')


for i in range(10):
    if isGray:
        plt.imshow(x_train[i, 0, :, :].astype(np.uint8), cmap=plt.get_cmap('gray'))
    else:
        tmp_img = x_train[i].transpose(1,2,0).astype(np.uint8)
        plt.imshow(tmp_img)
    plt.show()

print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('x_validation.shape: ', x_validation.shape)
print('y_validation.shape: ', y_validation.shape)
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)

del d