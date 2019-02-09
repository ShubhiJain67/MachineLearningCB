# Import the libraries
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import pandas as pd
import cv2
import skimage.io as io

#Convert lists to numpy arrays
mylist = [[1, 2], [3, 4]]
arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
# print arr
#  print arr.dtype

# print mylist
# print arr
# print type(mylist), type(arr)
# print arr.dtype, arr.shape

# Create New numpy arrays
arr1 = np.zeros((5, 5))
# print arr1

arr1 = np.eye(5)
# print arr1

# Copy numpy arrays
arr2 = arr1.copy()
# print arr2

arr3 = np.random.randint(low=-5, high=10, size=(5, 6))
# print arr3

# Shapes of numpy arrays and multi-dimensional arrays
# print arr3.shape, arr3.dtype
# print arr3.astype(np.float32).dtype

# Reshape, Transpose, Concatenate numpy arrays
arr4 = arr3.reshape((6, -1))
# print arr4
# print arr4.shape

# Transpose
# print arr4
# print arr4.T

A = np.random.randint(low=-5, high=10, size=(3, 3))
B = np.random.randint(low=-15, high=20, size=(3, 3))

print (A)
print (B)
print ('-'*30)


# C = np.concatenate((np.expand_dims(A, axis=0),
#                     np.expand_dims(B, axis=0)), axis=2)
# print C.shape

# Inverse, Prod, Add, Rank, Determinant...
C = A.dot(A)
D = B.dot(A)


# M = np.random.randint(low=-5, high=10, size=(5, 4))
# print np.linalg.pinv(M).dot(M)
# TODO: Read about pseudo-inverse and SVD (Singular Value Decomposition)

# A[2, :] = A[0, :]*2
# print A
# print np.linalg.matrix_rank(A)

print (np.linalg.det(A))
