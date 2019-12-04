"""
test file for the regression tree algorithm
"""
import numpy as np
from sklearn.datasets import load_boston

from RegressionTree import RegressionTree, BoostedTree

### BOSTON case

cols = [0, 1, 2, 4, 5, 6, 7, 9, 10]
features = boston_dataset = load_boston().feature_names[cols]
X, y = load_boston(return_X_y=True)
data = np.hstack((X[:, cols], y[:, np.newaxis]))
np.random.seed(666)
np.random.shuffle(data)
numel = 100
X = data[numel:, :-1]
y = data[numel:, -1]
test = data[0:numel, :-1]
correct_values = data[0:numel, -1]
print("Mean of labels: {}".format(np.mean(correct_values)))


# fit a single tree
reg = RegressionTree(max_depth=2)
reg.fit(X, y)
prediction = reg.predict(test)

reg.print_tree(features)

# fit boosted trees
bst = BoostedTree(max_depth=2, num_iter=100)
bst.fit(X, y)
bst_prediction = bst.predict(test)

rmse = lambda exact, guess: np.sqrt(np.mean((exact-guess)**2))

with np.printoptions(precision=2, floatmode="fixed"):
    print("Simple tree prediction:")
    print(prediction)
    print("Boosted tree prediction:")
    print(bst_prediction)
    print("Correct values: ")
    print(correct_values)
    print("Simple tree rmse: {:.2f}".format(rmse(correct_values, prediction)))
    print("Boosted tree rmse: {:.2f}".format(rmse(correct_values, bst_prediction)))
