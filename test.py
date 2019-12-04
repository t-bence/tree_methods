"""
test file for the regression tree algorithm
"""

import numpy as np
from sklearn.datasets import load_boston
import unittest

from RegressionTree import RegressionTree

### BOSTON case

cols = [0, 1, 2, 4, 5, 6, 7, 9, 10]
features = boston_dataset = load_boston().feature_names[cols]
X, y = load_boston(return_X_y=True)
data = np.hstack((X[:, cols], y[:, np.newaxis]))
np.random.shuffle(data)
numel = 8
X = data[numel:, :-1]
y = data[numel:, -1]
test = data[0:numel, :-1]
correct_values = data[0:numel, -1]
print("Mean of labels: {}".format(np.mean(correct_values)))


### generic things
reg = RegressionTree(max_depth=3)
reg.fit(X, y)

reg.print_tree(features)

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

class Tests(unittest.TestCase):
    """
    Test case for made up data and 2-level tree
    """
    def test_made_up_data_depth_2(self):
        X = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
        y = np.array([10, 8, 7, 1, 2, 4])
        reg2 = RegressionTree(max_depth=2)
        reg2.fit(X, y)
        prediction2 = list(reg2.predict(np.array([[0.4, 0.4], [0.6, 0.6], [0.8, 0.4], [0.2, 0.8]])))
        self.assertEqual(prediction2, list(np.array([1.50, 7.50, 4.00, 10.00])))

    def test_made_up_data_depth_3(self):
        X = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
        y = np.array([10, 8, 7, 1, 2, 4])
        reg3 = RegressionTree(max_depth=3)
        reg3.fit(X, y)
        prediction3 = list(reg3.predict(np.array([[0.4, 0.4], [0.6, 0.6], [0.8, 0.4], [0.2, 0.8]])))
        self.assertEqual(prediction3, list(np.array([2.00, 7.00, 4.00, 10.00])))

if __name__ == '__main__':
    unittest.main()
