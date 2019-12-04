"""
test file for the regression tree algorithm
"""

import numpy as np
from sklearn.datasets import load_boston

from RegressionTree import RegressionTree

### BOSTON case

boston_dataset = load_boston()
cols = [0, 1, 2, 4, 5, 6, 7, 9, 10]
X = boston_dataset.data[10:, cols]
y = boston_dataset.target[10:]

test = boston_dataset.data[0:10, cols]
correct_values = boston_dataset.target[0:10]

print("Mean of labels: {}".format(np.mean(y)))


### Simple case
# X = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
# y = np.array([10, 8, 7, 1, 2, 4])
# test = np.array([[0.4, 0.4], [0.6, 0.6], [0.8, 0.4], [0.2, 0.8]])
# correct_values = np.array([1.50, 7.50, 4.00, 10.00])

reg = RegressionTree(max_depth=2)
reg.fit(X, y)

reg.print_tree(boston_dataset.feature_names[cols])

prediction = reg.predict(test)

with np.printoptions(precision=2, floatmode="fixed"):
    print("Prediction:")
    print(prediction)
    print("Correct values: ")
    print(correct_values)

# cut_feature, cut_value = reg._best_split(X, y)
# print("cut_feature: {}, expected: 1".format(cut_feature))
# print("cut_value: {}, expected: ~0.5".format(cut_value))

# expected output: something like (1, 0.5)
