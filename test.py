"""
test file for the regression tree algorithm
"""

import numpy as np
from RegressionTree import RegressionTree

X = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
y = np.array([10, 8, 7, 1, 2, 4])

reg = RegressionTree(max_depth=2)
reg.fit(X, y)

reg.print_tree()

test = np.array([[0.4, 0.4], [0.6, 0.6], [0.8, 0.4], [0.2, 0.8]])

prediction = reg.predict(test)

with np.printoptions(precision=2, floatmode="fixed"):
    print("Prediction:")
    print(prediction)
    print("Expected: [1.50, 8.33, 4.00, 10.00]")

# cut_feature, cut_value = reg._best_split(X, y)
# print("cut_feature: {}, expected: 1".format(cut_feature))
# print("cut_value: {}, expected: ~0.5".format(cut_value))

# expected output: something like (1, 0.5)
