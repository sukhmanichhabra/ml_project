import numpy as np
from tqdm import trange

# --- Core Utility Functions (Activation and Loss) ---

def relu(Z):
    """ReLU activation function."""
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    """Derivative of ReLU for backpropagation."""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    """Softmax activation function for the output layer."""
    # Stability fix: subtract max for numerical stability
    exp_z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(Y_hat, Y):
    """Categorical Cross-Entropy Loss (cost function)."""
    m = Y.shape[0]
    # Small epsilon to avoid log(0)
    epsilon = 1e-10
    cost = -np.sum(Y * np.log(Y_hat + epsilon)) / m
    return cost

def softmax_cross_entropy_backward(Y_hat, Y):
    """Derivative of softmax loss w.r.t Z (pre-activation of output layer)."""
    return Y_hat - Y

# --- Layer Classes ---

class Dense:
    """A standard fully connected (Dense) layer."""
    def __init__(self, n_input, n_output, activation='relu', lambda_param=0.0):
        self.n_input = n_input
        self.n_output = n_output
        self.activation_name = activation
        self.lambda_param = lambda_param # Regularization parameter
        
        # He initialization (suitable for ReLU)
        limit = np.sqrt(2.0 / n_input)
        self.W = np.random.randn(n_input, n_output) * limit
        self.b = np.zeros((1, n_output))
        
        # Store for backpropagation
        self.Z = None
        self.A_prev = None
        
        # Select activation functions
        self.activation = relu if activation == 'relu' else softmax
        self.activation_backward = relu_backward if activation == 'relu' else softmax_cross_entropy_backward

    def forward(self, A_prev):
        """Calculates Z = A_prev * W + b and A = activation(Z)."""
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA, Y=None):
        """Calculates dW, db, and dA_prev, including L2 regularization."""
        m = self.A_prev.shape[0]
        
        # 1. Compute dZ based on layer type
        if self.activation_name == 'softmax':
            dZ = self.activation_backward(self.A, Y)
        else:
            dZ = self.activation_backward(dA, self.Z)
        
        # 2. Compute gradients dW and db
        dW = (1 / m) * np.dot(self.A_prev.T, dZ)
        db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
        
        # 3. Add L2 Regularization term to dW (Weight Decay)
        if self.lambda_param > 0:
            dW += (self.lambda_param / m) * self.W 
        
        # 4. Compute dA_prev for the next (previous) layer's backward pass
        dA_prev = np.dot(dZ, self.W.T)
        
        return dA_prev, dW, db
    
    def regularization_loss(self):
        """Calculates L2 regularization cost for this layer."""
        if self.lambda_param > 0:
            return (self.lambda_param / 2) * np.sum(self.W**2)
        return 0

class Dropout:
    """A custom Dropout layer for regularization."""
    def __init__(self, rate=0.5):
        self.rate = rate
        # Scale to ensure expected value remains the same during training
        self.scale = 1 / (1 - rate)
        self.mask = None

    def forward(self, A, training=True):
        """Applies dropout during training, scales output during testing."""
        if not training:
            return A
        
        # Generate mask: 1 with probability 1 - rate, 0 otherwise
        self.mask = (np.random.rand(*A.shape) < (1 - self.rate)) * self.scale
        return A * self.mask

    def backward(self, dA):
        """Applies the same mask to the gradient during backprop."""
        return dA * self.mask

# --- Sequential Model Class ---

class Sequential:
    """The main container for the custom neural network."""
    def __init__(self, input_shape=None):
        self.layers = []
        self.input_shape = input_shape
        
    def add(self, layer):
        """Adds a layer (Dense, Dropout) to the model."""
        if self.input_shape and isinstance(layer, Dense):
            # The first Dense layer automatically receives the input shape
            if not hasattr(layer, 'W') or layer.W.shape[0] != self.input_shape: 
                # Re-initialize only if the shape is wrong or not yet set
                layer.__init__(self.input_shape, layer.n_output, layer.activation_name, layer.lambda_param)
            self.input_shape = layer.n_output
        self.layers.append(layer)

    def forward_propagation(self, X, training=True):
        """Performs forward pass through all layers."""
        A = X
        for layer in self.layers:
            if isinstance(layer, Dense):
                A = layer.forward(A)
            elif isinstance(layer, Dropout):
                A = layer.forward(A, training)
        return A # Y_hat

    def backward_propagation(self, Y_hat, Y):
        """Performs backpropagation and collects gradients."""
        gradients = []
        dA = None
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if isinstance(layer, Dense):
                is_output_layer = (i == len(self.layers) - 1)
                
                if is_output_layer:
                    dA_prev, dW, db = layer.backward(dA, Y=Y)
                else:
                    dA_prev, dW, db = layer.backward(dA)
                
                dA = dA_prev
                gradients.insert(0, (dW, db))
                
            elif isinstance(layer, Dropout):
                dA = layer.backward(dA)

        return gradients

    def update_parameters(self, gradients, learning_rate, clip_value=1.0):
        """Updates weights and biases using SGD with Gradient Clipping."""
        param_index = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                dW, db = gradients[param_index]
                
                # --- Gradient Clipping ---
                if clip_value is not None:
                    dW = np.clip(dW, -clip_value, clip_value)
                    db = np.clip(db, -clip_value, clip_value)

                layer.W -= learning_rate * dW
                layer.b -= learning_rate * db
                param_index += 1

    def calculate_cost(self, Y_hat, Y):
        """Calculates the total cost (Cross-Entropy + Regularization)."""
        m = Y.shape[0]
        ce_loss = cross_entropy_loss(Y_hat, Y)
        
        reg_loss = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                reg_loss += layer.regularization_loss()
        
        return ce_loss + (reg_loss / m)

    def fit(self, X, Y, learning_rate=0.01, n_iters=100, batch_size=32, lambda_param=0.01, clip_value=1.0, verbose=True):
        """Trains the model using Mini-Batch Stochastic Gradient Descent."""
        m = X.shape[0]
        
        # Set L2 lambda for all Dense layers
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.lambda_param = lambda_param
        
        if verbose:
            print(f"Starting MLP training: {n_iters} epochs, LR={learning_rate}, L2 Lambda={lambda_param}, Clip={clip_value}")
            
        for epoch in trange(n_iters, desc="Training MLP"):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            
            epoch_cost = 0
            num_batches = 0
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                Y_batch = Y_shuffled[i:i + batch_size]
                
                Y_hat = self.forward_propagation(X_batch, training=True)
                
                epoch_cost += self.calculate_cost(Y_hat, Y_batch) * X_batch.shape[0] # Scale cost by batch size
                num_batches += 1
                
                gradients = self.backward_propagation(Y_hat, Y_batch)
                
                self.update_parameters(gradients, learning_rate, clip_value)

            if verbose and (epoch + 1) % 50 == 0: 
                print(f"Epoch {epoch + 1}/{n_iters} | Avg Epoch Cost: {epoch_cost / m:.4f}")

    def predict(self, X):
        """Generates class predictions."""
        Y_hat = self.forward_propagation(X, training=False)
        return np.argmax(Y_hat, axis=1)
