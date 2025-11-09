import numpy as np
from collections import Counter
import sys # Added for safe print/warnings

# --- Node Class (No changes needed) ---
class Node:
    """Represents a node in the decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None, info_gain=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # This will store the predicted class (0-18)
        self.info_gain = info_gain

    def is_leaf_node(self):
        """Check if the node is a leaf."""
        return self.value is not None

# --- Decision Tree Classifier (Corrected for Multi-Class) ---
class DecisionTreeClassifier:
    """Decision Tree Classifier using Gini Impurity from scratch."""
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        if min_samples_split <= 1:
            raise ValueError("min_samples_split must be >= 2.")
        if max_depth <= 0:
             raise ValueError("max_depth must be positive.")
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self._tree_n_features = None

    def fit(self, X, y):
        """Build the decision tree."""
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")
        
        if X.shape[0] < self.min_samples_split:
             self.root = Node(value=self._most_common_label(y))
             self._tree_n_features = X.shape[1]
             return 

        n_total_features = X.shape[1]
        self._tree_n_features = n_total_features
        
        self.n_feats = n_total_features if self.n_feats is None or self.n_feats > n_total_features else self.n_feats
        
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Recursively build the tree node by node."""
        n_samples, n_features_in_data = X.shape
        unique_labels = np.unique(y)
        n_labels = len(unique_labels)

        # Stopping criteria
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features_in_data, self.n_feats, replace=False)
        
        best_feat, best_thresh, best_gain = self._best_criteria(X, y, feat_idxs)

        if best_gain <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child, info_gain=best_gain)

    def _best_criteria(self, X, y, feat_idxs):
        """Find the best split (feature index, threshold) by maximizing Gini gain."""
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._gini_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh, best_gain

    def _gini_impurity(self, y):
        """
        Calculate Gini impurity for a set of labels.
        --- THIS IS THE CORE MULTI-CLASS FIX ---
        """
        n_samples = len(y)
        if n_samples == 0: 
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        gini = 1.0 - np.sum(probabilities**2)
        return gini

    def _gini_gain(self, y, X_column, split_thresh):
        """Calculate Information Gain based on Gini impurity reduction."""
        parent_gini = self._gini_impurity(y)
        
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0: 
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        
        gini_l, gini_r = self._gini_impurity(y[left_idxs]), self._gini_impurity(y[right_idxs])
        weighted_child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r
        
        gini_gain = parent_gini - weighted_child_gini
        return gini_gain

    def _split(self, X_column, split_thresh):
        """Split a feature column into two arrays of indices based on a threshold."""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        """
        Find the most frequent label in a set.
        --- THIS IS THE CORE MULTI-CLASS FIX ---
        """
        if len(y) == 0:
            return 0 
        
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """Predict class labels for new data by traversing the tree."""
        if self.root is None: 
            raise RuntimeError("Model has not been fitted yet.")
        
        X = np.array(X)
        
        if self._tree_n_features is not None and X.shape[1] != self._tree_n_features:
             raise ValueError(f"Input feature dimension mismatch: Expected {self._tree_n_features}, got {X.shape[1]}")

        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        return predictions.astype(int)

    def _traverse_tree(self, x, node):
        """Recursively navigate the tree for a single data point's prediction."""
        if node.is_leaf_node():
            return node.value
        
        if node.feature_idx >= len(x):
             print(f"Warning: Feature index {node.feature_idx} out of bounds.", file=sys.stderr)
             return 0 

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# --------------------------------------------------------------------------
# --- Random Forest Classifier (Corrected for Multi-Class) ---
class RandomForestClassifier:
    """Random Forest Classifier from scratch (uses the DecisionTreeClassifier above)."""
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        if n_trees <= 0:
            raise ValueError("n_trees must be positive.")
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []
        self._forest_n_features = None

    def fit(self, X, y):
        """Train the random forest by building multiple decision trees."""
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        self.trees = []
        n_samples, n_features_total = X.shape
        self._forest_n_features = n_features_total

        num_feats_for_tree = self.n_feats
        if num_feats_for_tree is None:
             num_feats_for_tree = int(np.sqrt(n_features_total))
        num_feats_for_tree = max(1, min(num_feats_for_tree, n_features_total)) 

        print(f"Fitting Random Forest: {self.n_trees} trees, max_depth={self.max_depth}, min_split={self.min_samples_split}, features_per_split={num_feats_for_tree}...")
        
        for i in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree = DecisionTreeClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=num_feats_for_tree
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            if (i + 1) % max(1, self.n_trees // 10) == 0 or i == self.n_trees - 1:
                 print(f"  Tree {i+1}/{self.n_trees} fitted.")

        print("Random Forest fitting complete.")

    def _bootstrap_sample(self, X, y):
        """Create a random sample of the data with replacement (same size as original)."""
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y_votes):
        """
        Find the majority vote among predictions from different trees.
        --- THIS IS THE CORE MULTI-CLASS FIX ---
        """
        if len(y_votes) == 0:
             print("Warning: Received empty array for voting.", file=sys.stderr)
             return 0
        
        counter = Counter(y_votes)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """Predict labels by aggregating (majority voting) predictions from all trees."""
        if not self.trees:
             raise RuntimeError("Model has not been fitted yet.")
        
        X = np.array(X)
        
        if self._forest_n_features is not None and X.shape[1] != self._forest_n_features:
             raise ValueError(f"Input feature dimension mismatch: Expected {self._forest_n_features}, got {X.shape[1]}")

        print(f"Predicting labels for {X.shape[0]} samples using Random Forest ({self.n_trees} trees)...")
        
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds_transposed = tree_preds.T
        y_pred = [self._most_common_label(sample_preds) for sample_preds in tree_preds_transposed]

        return np.array(y_pred)

