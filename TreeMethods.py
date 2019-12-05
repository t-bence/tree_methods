"""
Regression Tree algorithm
"""
import unittest
import numpy as np
from anytree import NodeMixin, RenderTree

class AdaBoostedClassifier:
    """
    A boosted regressor tree ensemble object
    """

    def __init__(self, num_iter=3):
        """ Construct a boosted tree model """
        self.trees = []
        self.max_depth = 1  # use stumps
        self.num_iter = num_iter
        self.tree_weights = []

    def fit(self, X, y):
        """ Fit data and create model """
        data_weights = np.ones_like(y) / len(y)  # start with equal weights
        for _ in range(self.num_iter):
            tree = RegressorTree(max_depth=self.max_depth)
            tree.fit(X, y)
            prediction = tree.predict(X) >= 0.5
            mistaken = y == (~prediction)
            self.trees.append(tree)
            total_error = np.sum(data_weights * mistaken) # weights of incorrect elements
            tree_weight = 1/2 * np.log((1 - total_error) / (total_error + 1e-10) + 1e-10)
            self.tree_weights.append(tree_weight)
            data_weights[mistaken] = data_weights[mistaken] * np.exp(tree_weight)
            data_weights[~mistaken] = data_weights[~mistaken] * np.exp(-tree_weight)
            data_weights = data_weights / np.sum(data_weights)
            # generate new data set from sampling with replacement
            inds = np.random.choice(y.shape[0], size=y.shape[0], replace=True, p=data_weights)
            X = X[inds, :]
            y = y[inds]
        return self

    def predict(self, X):
        """ Predict values using the model """
        prediction = np.zeros((X.shape[0],))
        for tree, tree_weight in zip(self.trees, self.tree_weights):
            prediction += tree_weight * tree.predict(X)
        return prediction >= 0.5

class BoostedRegressor:
    """
    A boosted regressor tree ensemble object
    """

    def __init__(self, max_depth=3, num_iter=3):
        """ Construct a boosted tree model """
        self.trees = []
        self.max_depth = max_depth
        self.num_iter = num_iter

    def fit(self, X, y):
        """ Fit data and create model """
        residual = y
        for _ in range(self.num_iter):
            tree = RegressorTree(max_depth=self.max_depth)
            tree.fit(X, residual)
            residual = residual - tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        """ Predict values using the model """
        prediction = np.zeros((X.shape[0],))
        for tree in self.trees:
            prediction += tree.predict(X)
        return prediction


class RegressorTree:
    """
    A self-made regression tree to learn about how these things work
    A nodal decision should always look like x_j <= a:
    left (0th child) if True, right (1st child) if False
    """

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Train the model with features in X and label in y
        """
        num_rows = X.shape[0]
        # create root node
        root = RegressorNode(indices=np.asarray([True] * num_rows))
        for _ in range(self.max_depth):
            for leaf in root.leaves:
                leaf.split(X, y)
        self.tree = root
        return self

    def predict(self, X):
        """
        Predict the values for the input in X
        """
        # split data set
        assert self.tree, "The tree must exist, call fit before predict!"
        y = np.zeros((X.shape[0], ))
        # propagate the values down the leaves
        for leaf in self.tree.leaves:
            indices = np.array([True] * X.shape[0])  # initialize dummy index
            for ancestor in NodeMixin.iter_path_reverse(leaf):  # iteration begins at the leaf node
                if ancestor.is_leaf:
                    left = ancestor.left
                else:
                    left_indices = X[:, ancestor.split_feature] <= ancestor.split_value
                    if left:
                        indices &= left_indices
                    else:
                        indices &= ~left_indices
                    left = ancestor.left
            y[indices] = leaf.value
        return y

    def print_tree(self, feature_names=None):
        """
        Print a graphical representation of a tree
        """
        for pre, _, node in RenderTree(self.tree):
            if feature_names is not None:
                if node.is_leaf:
                    treestr = pre + str(node)
                else:
                    treestr = pre + "{} <= {:.4f}?".format(feature_names[node.split_feature],
                                                           node.split_value)
            else:
                treestr = pre + str(node)
            print(treestr.ljust(6))


class Node(NodeMixin):
    """
    A base class to store nodes - should not be used directly.
    Use RegressorNode or ClassifierNode!
    """
    def __init__(self, indices=True, parent=None, left=True, value=None):
        self.parent = parent
        self.indices = indices
        self.split_feature = None
        self.split_value = None
        self.value = value
        self.left = left

    def split(self, X, y):
        """ Split a node to minimise a target in the two resulting nodes """
        assert not self.children, "Can't split a node that already has children"
        if self.indices.sum() <= 1: # don't split if there is one element only
            return None
        # find out best split: search all directions but only at each tenth percentile of values
        percentiles = np.percentile(X, np.arange(10, 100, 10), axis=0)
        split_target = np.zeros_like(percentiles)
        for val in range(split_target.shape[0]):
            for feat in range(split_target.shape[1]):
                limit = percentiles[val, feat]
                left_indices = X[self.indices, feat] <= limit
                split_target[val, feat] = self._split_target(y[self.indices], left_indices)
        value_index, self.split_feature = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
            np.argmin(split_target),
            split_target.shape
        )
        self.split_value = percentiles[value_index, self.split_feature]
        # indices in the <= part (left node)
        left_indices = X[:, self.split_feature] <= self.split_value
        self.children = self.make_children(y, left_indices)

        return (self.split_feature, self.split_value)

    def make_children(self, y, left_indices):
        """ Return the children """
        raise NotImplementedError

    def _split_target(self, y, indices):
        """ Calculate a splitting target to be minimised """
        raise NotImplementedError

    def __str__(self):
        return "<Node>"


class RegressorNode(Node):
    """ A node to be used for regression tasks """

    def make_children(self, y, left_indices):
        """ Calculate leaf values and return children """
        leaf_values = []
        for left_index in (left_indices, ~left_indices):
            assert (self.indices & left_index).any(), "empty leaf found"
            leaf_values.append(np.mean(y[self.indices & left_index]))
        return [
            RegressorNode(indices=(self.indices & left_indices), parent=self,
                          left=True, value=leaf_values[0]),
            RegressorNode(indices=(self.indices & (~left_indices)), parent=self,
                          left=False, value=leaf_values[1])
        ]

    def _split_target(self, y, indices):
        """ Calculate a splitting target: sum of variances
        """
        var = np.var(y[indices]) if indices.any() else np.Inf
        var += np.var(y[~indices]) if (~indices).any() else np.Inf
        # Inf is used here to force not to split in a way that nothing remains on one side
        return var

    def __str__(self):
        if self.is_leaf:
            return "{:.2f}".format(self.value)
        return "x_{} <= {:.2f}?".format(self.split_feature, self.split_value)


class ClassifierNode(Node):
    """ A node to be used for regression tasks """

    def make_children(self, y, left_indices):
        """ Calculate leaf values and return children """
        leaf_values = []
        assert (self.indices & left_indices).any(), "empty leaf found"
        assert (self.indices & ~left_indices).any(), "empty leaf found"

        # find the most common boolean element on the left side; the right side is the opposite
        left_value = np.mean(y[self.indices & left_indices]) >= 0.5
        leaf_values = [left_value, ~left_value]

        return [
            ClassifierNode(indices=(self.indices & left_indices), parent=self,
                           left=True, value=leaf_values[0]),
            ClassifierNode(indices=(self.indices & (~left_indices)), parent=self,
                           left=False, value=leaf_values[1])
        ]

    def _split_target(self, y, indices):
        """ Calculate a splitting target: Gini index """
        if not indices.any() or not (~indices).any():
            return np.Inf

        # for a leaf: gini = 1 - p^2(True) - p^2(False)
        left = 1.0 - (np.sum(y[indices]) / np.sum(indices)) ** 2
        left -= (np.sum(~y[indices]) / np.sum(indices)) ** 2

        right = 1.0 - (np.sum(y[~indices]) / np.sum(~indices)) ** 2
        right -= (np.sum(~y[~indices]) / np.sum(~indices)) ** 2

        # weighted average
        gini = (left * np.sum(indices) + right * np.sum(~indices)) / len(indices)
        return gini

    def __str__(self):
        if self.is_leaf:
            return "{}".format(self.value)
        return "x_{} <= {:.2f}?".format(self.split_feature, self.split_value)


class Tests(unittest.TestCase):
    """
    Test case for made up data and 2-level tree
    """
    def test_best_split(self):
        """ Test where splitting is done """
        test_X = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
        test_y = np.array([10, 8, 7, 1, 2, 4])
        node = RegressorNode(indices=np.array([True]*6))
        cut_feature, cut_value = node.split(test_X, test_y)
        self.assertEqual(cut_feature, 1)
        self.assertTrue(cut_value >= 0.4)
        self.assertTrue(cut_value < 0.6)

    def test_classifier(self):
        """ Test the classifier following youtube.com/watch?v=LsK-xG1cLYA """
        test_X = np.array([[True, True, 205], [False, True, 180], [True, False, 210],
                           [True, True, 167], [False, True, 156], [False, True, 125],
                           [True, False, 168], [True, True, 172]])
        test_y = np.array([True, True, True, True, False, False, False, False])
        test_reg = RegressorTree(max_depth=1)
        test_reg.fit(test_X, test_y)
        # test_reg.print_tree(['ChPn', 'BlAr', 'PaWe'])
        self.assertTrue(
            list(test_reg.predict(np.array([[True, False, 180], [True, False, 172]]))),
            [True, False])

    def test_AdaBoost_classifier(self):
        """ Test the classifier following youtube.com/watch?v=LsK-xG1cLYA """
        test_X = np.array([[True, True, 205], [False, True, 180], [True, False, 210],
                           [True, True, 167], [False, True, 156], [False, True, 125],
                           [True, False, 168], [True, True, 172]])
        test_y = np.array([True, True, True, True, False, False, False, False])
        test_ada = AdaBoostedClassifier(num_iter=1)
        test_ada.fit(test_X, test_y)
        pred_ada = test_ada.predict(np.array([[True, False, 180], [True, False, 172]]))
        # test_ada.trees[0].print_tree(['ChPn', 'BlAr', 'PaWe'])
        self.assertEqual(list(pred_ada), [True, False])

    def test_simplest_case(self):
        """ Test a simplistic case """
        test_X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        test_y = np.array([1, 2, 3, 4])
        test_reg = RegressorTree(max_depth=2)
        test_reg.fit(test_X, test_y)
        test_pred = test_reg.predict(np.array([[-1, -1], [2, -1], [-1, 2], [2, 2]]))
        self.assertEqual(list(test_pred), [1., 2., 3., 4.])

    def test_made_up_data_depth_2(self):
        """ Test with 2-level tree """
        test_X = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
        test_y = np.array([10, 8, 7, 1, 2, 4])
        test_reg = RegressorTree(max_depth=2)
        test_reg.fit(test_X, test_y)
        test_pred = test_reg.predict(np.array([[0.4, 0.4], [0.6, 0.6], [0.8, 0.4], [0.2, 0.8]]))
        self.assertEqual(list(test_pred), [1.50, 7.50, 4.00, 10.00])

    def test_made_up_data_depth_3(self):
        """ Test with 3-level tree """
        test_X = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
        test_y = np.array([10, 8, 7, 1, 2, 4])
        test_reg = RegressorTree(max_depth=3)
        test_reg.fit(test_X, test_y)
        test_pred = test_reg.predict(np.array([[0.4, 0.4], [0.6, 0.6], [0.8, 0.4], [0.2, 0.8]]))
        self.assertEqual(list(test_pred), [2.00, 7.00, 4.00, 10.00])

if __name__ == '__main__':
    unittest.main()
