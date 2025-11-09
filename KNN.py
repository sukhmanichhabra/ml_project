import numpy as np
from collections import Counter

class KNN:
    """
    k-Nearest Neighbors (k-NN) classifier implemented entirely from scratch.
    
    It supports multi-class classification and uses Euclidean distance.
    NOTE: This model is a 'lazy learner' as it does not explicitly train weights.
    """
    def __init__(self, k=5):
        # k is the number of neighbors to consider
        self.k = k

    def fit(self, X_train, y_train):
        """
        Fitting just means storing the training data.
        """
        if len(np.unique(y_train)) < 2:
            raise ValueError("Training data must contain at least two classes.")
            
        self.X_train = X_train
        self.y_train = y_train
        print(f"Custom KNN fitted with {len(X_train)} samples (k={self.k}).")

    def _euclidean_distance(self, x1, x2):
        """
        Calculates the Euclidean distance (L2 norm) between two vectors.
        """
        return np.sqrt(np.sum((x1 - x2)**2))

    def _predict(self, x):
        """
        Helper function to predict the class of a single sample 'x'.
        """
        # 1. Calculate distances from 'x' to all training samples
        distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
        
        # 2. Get the indices of the k smallest distances
        k_indices = np.argsort(distances)[:self.k]
        
        # 3. Extract the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # 4. Determine the most common class (majority vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]

    def predict(self, X_test):
        """
        Predicts the class label for each sample in the test set X_test.
        """
        print(f"Predicting for {len(X_test)} samples using Custom KNN...")
        # Apply _predict to every sample in X_test
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
