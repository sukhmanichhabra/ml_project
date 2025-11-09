import numpy as np
import random

class SimpleMLP:
    """Simple Multi-Layer Perceptron (1 hidden layer) for MULTI-CLASS classification."""
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01, n_iters=1000, verbose=False, activation='relu'):
        if n_input <= 0 or n_hidden <= 0 or n_output <= 0:
            raise ValueError("Layer sizes must be positive.")
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output # This MUST be 19 for your dataset
        self.lr = learning_rate
        self.n_iters = n_iters
        self.verbose = verbose
        self.activation_type = activation.lower()
        if self.activation_type not in ['sigmoid', 'relu']:
             raise ValueError("Activation must be 'sigmoid' or 'relu'.")

        np.random.seed(42) 
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2/n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2/n_hidden)
        self.b2 = np.zeros((1, n_output))
        
        self.cost_history = []

    # --- Activation Functions ---
    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_prime(self, Z):
        return (Z > 0).astype(int)

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_prime(self, Z):
        s = self._sigmoid(Z)
        return s * (1 - s)

    # --- FIX 1: Added Softmax for Multi-Class Output ---
    def _softmax(self, Z):
        """Softmax activation function (numerically stable)."""
        # Shift Z by max(Z) for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True)) 
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # --- Forward Propagation ---
    def _forward_propagation(self, X):
        Z1 = X.dot(self.W1) + self.b1
        if self.activation_type == 'relu':
            A1 = self._relu(Z1)
        else: # 'sigmoid'
            A1 = self._sigmoid(Z1)

        Z2 = A1.dot(self.W2) + self.b2
        # --- FIX 2: Using Softmax for Output Layer ---
        A2 = self._softmax(Z2) 
        
        cache = (Z1, A1, Z2, A2)
        return A2, cache

    # --- Cost Function (Loss) ---
    # --- FIX 3: Corrected Cost Function to Categorical Cross-Entropy ---
    def _compute_cost(self, A2, Y):
        """
        Categorical Cross-Entropy Loss for multi-class.
        A2 is (m, n_output) matrix of probabilities
        Y is (m,) array of true class indices (0, 1, ..., 18)
        """
        m = Y.shape[0]
        # A2[range(m), Y] selects the probabilities of the *correct* class for each sample
        log_probs = -np.log(A2[range(m), Y] + 1e-10) # Add epsilon for stability
        cost = np.sum(log_probs) / m
        return np.squeeze(cost)

    # --- Backward Propagation ---
    def _backward_propagation(self, cache, X, Y):
        (Z1, A1, Z2, A2) = cache
        m = X.shape[0]

        # Convert scalar labels Y (e.g., [0, 2, 1]) to One-Hot Encoding (m x n_output)
        Y_one_hot = np.zeros((m, self.n_output))
        Y_one_hot[range(m), Y] = 1 # Y contains the indices (0-18)

        # --- FIX 4: Corrected Backpropagation for Softmax/Cross-Entropy ---
        # The derivative of Cross-Entropy Loss w.r.t Z2 (pre-softmax)
        # is simply (Probabilities - True_One_Hot)
        dZ2 = A2 - Y_one_hot
        
        # Gradients for Layer 2
        dW2 = (1/m) * A1.T.dot(dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # Gradients for Layer 1
        dA1 = dZ2.dot(self.W2.T)
        
        # Derivative of the hidden layer activation
        if self.activation_type == 'relu':
            dZ1 = dA1 * self._relu_prime(Z1)
        else: # 'sigmoid'
            dZ1 = dA1 * self._sigmoid_prime(Z1)

        # Gradients for Layer 1
        dW1 = (1/m) * X.T.dot(dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # --- Update Parameters ---
    def _update_parameters(self, grads):
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]

    # --- Training and Prediction ---
    def fit(self, X, y):
        """Train the MLP."""
        X = np.array(X)
        Y = np.array(y) 

        print(f"Starting training with {self.n_iters} iterations (Activation: {self.activation_type}/Softmax)...")

        for i in range(self.n_iters):
            A2, cache = self._forward_propagation(X)
            cost = self._compute_cost(A2, Y)
            self.cost_history.append(cost)
            grads = self._backward_propagation(cache, X, Y)
            self._update_parameters(grads)

            if self.verbose and (i % max(1, self.n_iters // 10) == 0 or i == self.n_iters - 1):
                print(f"Iteration {i}: Cost = {cost:.6f}")

        print(f"Training complete. Final Cost = {self.cost_history[-1]:.6f}")

    def predict(self, X):
         """Predict labels for input X."""
         if self.W1 is None: raise RuntimeError("Model has not been fitted yet.")
         X = np.array(X)
         if X.shape[1] != self.n_input:
             raise ValueError(f"Input feature dimension mismatch: Expected {self.n_input}, got {X.shape[1]}")

         # A2 is the probability matrix (m x n_output) after Softmax
         A2, _ = self._forward_propagation(X)
         
         # The prediction is the index with the highest probability
         # Returns integer class indices (0 to 18)
         return np.argmax(A2, axis=1)