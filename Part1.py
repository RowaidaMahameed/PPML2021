from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
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

def Tree_Train(X, y, depth, v):

    if depth == 29:
        print(y)

    if len(list(np.unique(y, axis=0)))==1: # check stop condition
        v.leaf_value = np.argmax(np.unique(y, axis=0))

    elif depth == maximal_depth: # check if got to maximal depth
        print(y, depth)
        sum = np.zeros(y[0].shape)
        for i in range(len(X)):
            sum += y[i]
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
                    right[(i_feature, threshold)] += I0(x[i_feature] - threshold) * y[i_sample]
                    left[(i_feature, threshold)] += I0(threshold - x[i_feature]) * y[i_sample]
        v.feature, v.threshold = Gini(X, right, left)

        # split the data
        X_right, X_left, y_right, y_left = [], [], [], []
        for i in range(len(X)):
            x = X[i]
            if I0(x[v.feature] - v.threshold):
                X_right.append(x)
                y_right.append(y[i])
            else:
                X_left.append(x)
                y_left.append(y[i])
        X_right, X_left, y_right, y_left = np.array(X_right), np.array(X_left), np.array(y_right), np.array(y_left)
        if depth==29:
            print(X_right, X_left, y_right, y_left, v.feature, v.threshold)
        # build right and left childs
        v.right, v.left = Node(), Node()
        Tree_Train(X_right, y_right, depth + 1, v.right)
        Tree_Train(X_left, y_left, depth + 1, v.left)

def Tree_Predict(v, x):
    if v.leaf_value != None:
        return v.leaf_value
    else:
        if I0(x[v.feature] - v.threshold):
            return Tree_Predict(v.right, x)
        else:
            return Tree_Predict(v.left, x)


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
Tree_Train(np.array(x_train), np.array(y_train), 0, root)
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