import numpy as np

X_train = np.load("data/proccessed/X_train.npy")
X_test = np.load("data/proccessed/X_test.npy")

y_train = np.load("data/proccessed/y_train.npy").squeeze()
y_test = np.load("data/proccessed/y_test.npy").squeeze()