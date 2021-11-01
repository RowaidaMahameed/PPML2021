import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt
from src.training.decision_tree_training import DecisionTree
import numpy as np
import time
import math

start = time.monotonic()
maximal_depth = 30

class Node:
   def __init__(self):
      self.left = None
      self.right = None
      self.leaf_value = None
      self.feature = None
      self.threshold = None

   def print_tree(self, depth=0):
       if not(self.right) or not(self.left):
           print('%s[%s]' % ((depth * '  ', self.leaf_value)))
       else:
           print('%s[X%d < %.3f]' % ((depth * '  ', self.feature, self.threshold)))
           self.left.print_tree(depth + 1)
           self.right.print_tree(depth + 1)

def I0(x):
    return 1 if x >= 0 else 0

def get_polynomial_regression(data, target):
    min_poly_reg = None
    min_poly_feats = None
    mses = []
    degrees = [x for x in range(1, 6)]
    min_mse, min_deg = 1e10, 0
    x_train, x_test = data
    y_train, y_test = target
    for deg in degrees:
        print(f'Testing degree={deg}')
        transformed_samples = transform_independent_multivariate_sample(x_train, degree=deg)
        poly_reg = LinearRegression(fit_intercept=False)
        poly_reg.fit(transformed_samples, y_train)

        transformed_test_samples = transform_independent_multivariate_sample(x_test, degree=deg)
        poly_predict = poly_reg.predict(transformed_test_samples)
        poly_mse = mean_squared_error(y_test, poly_predict)
        mses.append(poly_mse)
        if min_mse > poly_mse:
            min_mse = poly_mse
            min_deg = deg
            min_poly_reg = poly_reg

    print('Best degree {} with MSE {}'.format(min_deg, min_mse))
    coeffs = min_poly_reg.coef_
    coeffs_list = [coeffs[i:i+(min_deg+1)] for i in range(0, len(coeffs), min_deg+1)]
    # return min_poly_reg, min_deg, min_mse
    return coeffs_list, min_deg


def phe(x):
    polynomial_coef, deg = get_polynomial_regression(_data, _target)
    p = polynomial_coef[0];
    return x*p[0];


def Gini(X, right, left):
    right_total_side, left_total_side, I_G = {}, {}, {}
    for f in range(len(X[0])):  # loop over features
        thresholds = []
        for i_threshold in X:  # loop over thresholds
            thresholds.append(i_threshold[f])
        if len(thresholds) == 2:
            thresholds.append((thresholds[0]+thresholds[1])/2)
        for t in thresholds:  # loop over thresholds
            right_total_side[(f, t)] = np.sum(right[(f, t)])
            left_total_side[(f, t)] = np.sum(left[(f, t)])
            s_right, s_left = 0, 0
            for l in range(len(right[(f, t)])):
                s_right += math.pow(right[(f, t)][l] / right_total_side[(f, t)], 2)
                s_left += math.pow(left[(f, t)][l] / left_total_side[(f, t)], 2)
            I_G[(f, t)] = (1 - s_right) * right_total_side[(f, t)] + (1 - s_left) * left_total_side[(f, t)]
    return min(I_G, key=I_G.get)

def Tree_Train(X, y, W, depth, v):

    if depth == maximal_depth: # check if got to maximal depth
        #print(y, depth)
        sum = np.zeros(y[0].shape)
        for i in range(len(X)):
            sum += W[i]*y[i]
        v.leaf_value = np.argmax(sum)

    else: # search for the best split and split
        # find the best attribute and threshold to split
        right, left = {}, {}
        for i_feature in range(len(X[0])): # loop over features
            thresholds = []
            for i_threshold in X:  # loop over thresholds
                thresholds.append(i_threshold[i_feature])
            if len(thresholds) == 2:
                thresholds.append((thresholds[0] + thresholds[1]) / 2)
            for threshold in thresholds:  # loop over thresholds
                right[(i_feature, threshold)] = 0
                left[(i_feature, threshold)] = 0
                for i_sample in range(len(X)):
                    x = X[i_sample]
                    right[(i_feature, threshold)] += W[i_sample] * phe(x[i_feature] - threshold) * y[i_sample]
                    left[(i_feature, threshold)] += W[i_sample] * phe(threshold - x[i_feature]) * y[i_sample]
        v.feature, v.threshold = Gini(X, right, left)

        # update the weights
        W_left, W_right = [], []
        for i in range(len(X)):
            W_left.append(W[i] * phe(threshold - x[i_feature]))
            W_right.append(W[i] * phe(x[i_feature] - threshold))
        W_left, W_right = np.array(W_left), np.array(W_right)

        # build right and left childs
        v.right, v.left = Node(), Node()
        Tree_Train(X, y, W_right, depth + 1, v.right)
        Tree_Train(X, y, W_left, depth + 1, v.left)

def Tree_Predict(v, x):
    if v.leaf_value != None: # v is leaf
        return v.leaf_value
    else:
        return phe(x[v.feature] - v.threshold) * Tree_Predict(v.right, x) +\
               phe(v.threshold - x[v.feature]) * Tree_Predict(v.left, x)


# load and prepare dataset
dataset = load_breast_cancer()
X = np.array(dataset['data'])
y = np.array(dataset['target'])
y = y.reshape(len(dataset['target']),1)
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
y = np.array(columnTransformer.fit_transform(y))
x_train, x_test, y_train, y_test = train_test_split(X, y)

# Train decision Tree on the data
root = Node()
W = list(np.ones(x_train.shape[0]))
Tree_Train(np.array(x_train), np.array(y_train), W, 0, root)
root.print_tree()

# Predict on Train set
c=0
for i in range(len(x_train)):
  if Tree_Predict(root, x_train[i]) == np.argmax(y_train[i]):
    c+=1
print("acc = ", c/len(x_train))

# Predict on Train set
c=0
for i in range(len(x_test)):
  if Tree_Predict(root, x_test[i]) == np.argmax(y_test[i]):
    c+=1
print("acc = ", c/len(x_test))

end = time.monotonic()
print(end - start)
