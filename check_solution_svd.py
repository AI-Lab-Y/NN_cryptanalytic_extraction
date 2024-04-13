'''
compute functionally equivalence using the method introduced by Carlini et al. at CRYPTO 2020
'''

import sys
import numpy as np
import numpy.linalg
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize
import pickle
from utils import align_deep_nns

# ------------load the real nn and extracted nn--------------

# load model
nn_shape = [8, 2, 2, 1]

real_nn_path = 'c1_{}_deep_nn'.format(len(nn_shape) - 2)
extracted_nn_path = '{}_deep_nn'.format(len(nn_shape) - 2)
for i in range(len(nn_shape)):
    real_nn_path = real_nn_path + '_' + str(nn_shape[i])
    extracted_nn_path = extracted_nn_path + '_' + str(nn_shape[i])
real_nn_path = real_nn_path + '.npz'
extracted_nn_path = extracted_nn_path + '.npz'

real_model_path = './models/real_models/' + real_nn_path
fcn_real = np.load(real_model_path, allow_pickle=True)
ws_real, bs_real = fcn_real['arr_0'], fcn_real['arr_1']

extracted_model_path = './models/extracted_models/' + extracted_nn_path
fcn_extract = np.load(extracted_model_path, allow_pickle=True)
ws_extract, bs_extract = fcn_extract['arr_0'], fcn_extract['arr_1']

# ------------align the real nn and extracted nn-------------

print("Compute the matrix alignment for the SVD upper bound")

ws_real_align, bs_real_align = align_deep_nns(ws_real=ws_real, bs_real=bs_real,
                                                ws_extract=ws_extract, bs_extract=bs_extract,
                                                nn_shape=nn_shape)

A1, B1 = ws_real_align, bs_real_align
A2, B2 = ws_extract, bs_extract

# -----------------compute the max error-------------------

print("Finished alignment. Now compute the max error in the matrix.")
max_err = 0
for l in range(len(A1)):
    print("Matrix diff", np.sum(np.abs(A1[l] - A2[l])))
    print("Bias diff", np.sum(np.abs(B1[l] - B2[l])))
    max_err = max(max_err, np.max(np.abs(A1[l] - A2[l])))
    max_err = max(max_err, np.max(np.abs(B1[l] - B2[l])))

print("Number of bits of precision in the weight matrix",
      -np.log(max_err) / np.log(2))

# -----------------compute SVD upper bound-----------------

print("\nComputing SVD upper bound")
high = np.ones(A1[0].shape[0])
low = -np.ones(A1[0].shape[0])
input_bound = np.sum((high - low) ** 2) ** .5
prev_bound = 0
for i in range(len(A1)):
    largest_value = np.linalg.svd(A1[i] - A2[i])[1][0] * input_bound
    largest_value += np.linalg.svd(A1[i])[1][0] * prev_bound
    largest_value += np.sum((B1[i] - B2[i]) ** 2) ** .5
    prev_bound = largest_value
    print("\tAt layer", i, "loss is bounded by", largest_value)

print('Upper bound on number of bits of precision in the output through SVD', -np.log(largest_value) / np.log(2))