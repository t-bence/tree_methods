"""
Regression Tree algorithm
"""
import unittest
import numpy as np
from anytree import NodeMixin, RenderTree

class BoostedTree:
    """
    A boosted tree learer object
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
            tree = RegressionTree(max_depth=self.max_depth)
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


class RegressionTree:
    """
    A self-made regression tree to learn about how these things work...
    Tóth Bence, 2019. 12. 02.
    A nodal decision should always look like x_j <= a:
    left (0th child) if True, right (1st child) if False
    A node is a list of 3-tuples like: (j, a, True) meaning x_j <= a is True
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
        root = Node(indices=np.asarray([True] * num_rows))
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
        left = True
        for leaf in self.tree.leaves:
            indices = np.array([True] * X.shape[0])  # initialize dummy index
            for ancestor in NodeMixin.iter_path_reverse(leaf):
                if ancestor.split_value:
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
    A class to store nodes
    """
    def __init__(self, indices=True, parent=None, left=True, value=None):
        self.parent = parent
        self.indices = indices
        self.split_feature = None
        self.split_value = None
        self.value = value
        self.left = left

    def split(self, X, y):
        """
        Split a node to minimise the sum of the variances in the two resulting nodes
        """
        assert not self.children, "Can't split a node that already has children"
        if self.indices.sum() <= 1: # don't split if there is one element only
            return
        # data in the present node
        X_here = X[self.indices, :]
        y_here = y[self.indices]
        # find best split
        self.split_feature, self.split_value = self._best_split(X_here, y_here)
        # indices in the <= part (left node)
        left_indices = X[:, self.split_feature] <= self.split_value
        leaf_values = self._leaf_values(y, left_indices)
        self.children = [
            Node(indices=(self.indices & left_indices), parent=self,
                 left=True, value=leaf_values[0]),
            Node(indices=(self.indices & (~left_indices)), parent=self,
                 left=False, value=leaf_values[1])
        ]

    def prune(self):
        """
        Prune the tree: remove less useful branches based on cost-complexity quantification
        """
        pass

    def _leaf_values(self, y, left_indices):
        """
        Return the values of the two leaves at a terminal node: left is where x<=a, right is else,
        the value is the mean of the values contained in the partitions
        """
        leaf_values = []
        for left_index in (left_indices, ~left_indices):
            assert (self.indices & left_index).any(), "empty leaf found"
            leaf_values.append(np.mean(y[self.indices & left_index]))
        return leaf_values


    def _best_split(self, X, y):
        """
        When splitting a node, return a tuple of (feature, value) corresponding to the optimal
        split: feature is the number of the feature which has to be split and value is the
        optimal splitting value
        """

        # find out best split: search all directions but only at each tenth percentile of values
        percentiles = np.percentile(X, np.arange(10, 100, 10), axis=0)
        variances = np.zeros_like(percentiles)

        def var_sum(data, indices):
            """
            Calculate sum of variances in the two child nodes
            """
            var = np.var(data[indices], axis=0) if indices.any() else np.Inf
            var += np.var(data[~indices], axis=0) if (~indices).any() else np.Inf
            # Inf is used here to force not to split in a way that nothing remains on one side
            return var

        for val in range(variances.shape[0]):
            for feat in range(variances.shape[1]):
                limit = percentiles[val, feat]
                left_indices = X[:, feat] <= limit
                variances[val, feat] = var_sum(y, left_indices)
        value_index, feature = np.unravel_index(np.argmin(variances), variances.shape) # pylint: disable=unbalanced-tuple-unpacking

        return (feature, percentiles[value_index, feature])

    def __str__(self):
        if self.is_leaf:
            return "{:.2f}".format(self.value)
        return "x_{} <= {:.2f}?".format(self.split_feature, self.split_value)

class Tests(unittest.TestCase):
    """
    Test case for made up data and 2-level tree
    """
    def test_best_split(self):
        """ Test where splitting is done """
        X1 = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
        y1 = np.array([10, 8, 7, 1, 2, 4])
        node = Node()
        cut_feature, cut_value = node._best_split(X1, y1) # pylint: disable=protected-access
        self.assertEqual(cut_feature, 1)
        self.assertTrue(cut_value >= 0.4)
        self.assertTrue(cut_value < 0.6)

        # expected output: something like (1, 0.5)
    def test_made_up_data_depth_2(self):
        """ Test with 2-level tree """
        X2 = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
        y2 = np.array([10, 8, 7, 1, 2, 4])
        reg2 = RegressionTree(max_depth=2)
        reg2.fit(X2, y2)
        prediction2 = list(reg2.predict(np.array([[0.4, 0.4], [0.6, 0.6], [0.8, 0.4], [0.2, 0.8]])))
        self.assertEqual(prediction2, list(np.array([1.50, 7.50, 4.00, 10.00])))

    def test_made_up_data_depth_3(self):
        """ Test with 3-level tree """
        X3 = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.6], [0.3, 0.4], [0.5, 0.3], [0.7, 0.4]])
        y3 = np.array([10, 8, 7, 1, 2, 4])
        reg3 = RegressionTree(max_depth=3)
        reg3.fit(X3, y3)
        prediction3 = list(reg3.predict(np.array([[0.4, 0.4], [0.6, 0.6], [0.8, 0.4], [0.2, 0.8]])))
        self.assertEqual(prediction3, list(np.array([2.00, 7.00, 4.00, 10.00])))

if __name__ == '__main__':
    unittest.main()
