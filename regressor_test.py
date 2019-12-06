"""
test file for the regression tree algorithm
"""
import numpy as np
from TreeMethods import RegressorTree, GradientBoostedRegressor

### BOSTON case
"""
from sklearn.datasets import load boston
cols = [0, 1, 2, 4, 5, 6, 7, 9, 10]
features = boston_dataset = load_boston().feature_names[cols]
X, y = load_boston(return_X_y=True)
data = np.hstack((X[:, cols], y[:, np.newaxis]))
np.random.seed(666)
np.random.shuffle(data)
numel = 5
X = data[numel:, :-1]
y = data[numel:, -1]
test = data[0:numel, :-1]
correct_values = data[0:numel, -1]
print("Mean of labels: {}".format(np.mean(correct_values)))
"""

X = np.array([[1.6, 2, 0], [1.6, 1, 1], [1.5, 2, 1], [1.8, 0, 0],
              [1.5, 1, 0], [1.4, 2, 1]])
y = np.array([88, 76, 56, 73, 77, 57])
features = ['height', 'color', 'gender']
test = np.array([[1.6, 2, 0], [1.7, 2, 1]])
correct_values = np.array([88, 70])

# fit a single tree
reg = RegressorTree(max_depth=2)
reg.fit(X, y)
prediction = reg.predict(test)
reg.print_tree(features)

# fit boosted trees
bst = GradientBoostedRegressor(max_depth=2, num_iter=10)
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
