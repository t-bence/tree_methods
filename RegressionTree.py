"""
Regression Tree algorithm
"""

import numpy as np
from anytree import NodeMixin, RenderTree

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
        root.split(X, y)
        root.children[0].split(X, y)
        self.tree = root

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
            value = leaf.value
            for ancestor in NodeMixin.iter_path_reverse(leaf):
                if ancestor.split_value:
                    left_indices = X[:, ancestor.split_feature] <= ancestor.split_value
                    if left:
                        indices &= left_indices
                    else:
                        indices &= ~left_indices
                left = ancestor.left
            y[indices] = value
        return y

    def print_tree(self):
        """
        Print a graphical representation of a tree
        """
        for pre, _, node in RenderTree(self.tree):
            if node.is_leaf:  # final leaf
                treestr = u"{}{}{:.2f}".format(pre, node.left.__str__().ljust(6), node.value)
            else:
                treestr = u"{}x_{} <= {:.2f}".format(pre, node.split_feature, node.split_value)
            print(treestr.ljust(10))


class Node(NodeMixin):
    """
    A class to store nodes
    """
    def __init__(self, indices=None, parent=None, left=True, value=None):
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
        # data in the present node
        X_here = X[self.indices, :]
        y_here = y[self.indices]
        # find best split
        self.split_feature, self.split_value = self._best_split(X_here, y_here)
        # indices in the <= part (left node)
        left_indices = X[:, self.split_feature] <= self.split_value
        left_value, right_value = self._leaf_values(y, left_indices)
        self.children = [
            Node(indices=(self.indices & left_indices), parent=self, left=True, value=left_value),
            Node(indices=(self.indices & (~left_indices)), parent=self, left=False, value=right_value)
        ]
        return self.children

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
        left_value = np.mean(y[self.indices & left_indices]) if (self.indices & left_indices).any() else np.NaN
        right_value = np.mean(y[self.indices & ~left_indices]) if (self.indices & ~left_indices).any() else np.NaN
        return (left_value, right_value)


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
            var = np.var(data[indices], axis=0) if indices.any() else 0
            var += np.var(data[~indices], axis=0) if (~indices).any() else 0
            return var

        for val in range(variances.shape[0]):
            for feat in range(variances.shape[1]):
                limit = percentiles[val, feat]
                left_indices = X[:, feat] <= limit
                variances[val, feat] = var_sum(y, left_indices)
        value_index, feature = np.unravel_index(np.argmin(variances), variances.shape) # pylint: disable=unbalanced-tuple-unpacking

        return (feature, percentiles[value_index, feature])

    def __str__(self):
        if self.split_feature:
            return "Node: x_{} <= {}?".format(self.split_feature, self.split_value)
        else:
            return "{}: {}".format("Y" if self.left else "N", self.value)
        