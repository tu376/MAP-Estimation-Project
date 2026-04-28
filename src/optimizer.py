import numpy as np

class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01, epochs=1000, tol=1e-6):
        self.lr = learning_rate
        self.epochs = epochs
        self.tol = tol

    def minimize(self, theta_init, grad_func, data, **kwargs):
        """
        General Gradient Descent Algorithm.
        Goal: Find theta to minimize the loss function (Negative Log-Posterior).
        """
        theta = np.array(theta_init, dtype=float)
        
        for i in range(self.epochs):
            # Compute the gradient at the current point
            grad = grad_func(theta, data, **kwargs)
            
            # Update parameters (stepping in the opposite direction of the gradient)
            new_theta = theta - self.lr * grad
            
            # Check for convergence
            if np.linalg.norm(new_theta - theta) < self.tol:
                break
            theta = new_theta
            
        return theta


def grad_coin_flipping(theta, data, alpha=2, beta=2):
    """
    1. Coin Flipping (Bernoulli Likelihood + Beta Prior)
    """
    data = np.array(data)
    k = np.sum(data)   # Number of heads
    n = len(data)      # Total number of flips
    
    # Clip theta within (0, 1) to avoid log errors or division by zero
    theta = np.clip(theta, 1e-10, 1 - 1e-10)
    
    # Derivative of the Negative Log-Posterior
    term1 = (k + alpha - 1) / theta
    term2 = (n - k + beta - 1) / (1 - theta)
    return -(term1 - term2)

def grad_sensor_fusion(theta, data, sigma_likeli=1.0, mu_prior=0, sigma_prior=1.0):
    """
    2. Sensor Fusion (Gaussian Likelihood + Gaussian Prior)
    """
    data = np.array(data)
    # Derivative of the -Log Likelihood (Gaussian)
    grad_likeli = np.sum(theta - data) / (sigma_likeli**2)
    # Derivative of the -Log Prior (Gaussian)
    grad_prior = (theta - mu_prior) / (sigma_prior**2)
    
    return grad_likeli + grad_prior

def grad_ridge_regression(w, data, lmbda=0.1, sigma=1.0):
    """
    3. Ridge Regression (Linear Likelihood + Gaussian Prior)
    w: [w0, w1, ...] (intercept + weights)
    """
    X, y = data
    X = np.array(X)
    y = np.array(y)
    
    # Add a column of ones to account for the bias term (intercept w0)
    X_b = np.column_stack([np.ones(len(X)), X])
    
    # Ridge derivative formula: X.T @ (Xw - y) / sigma^2 + lambda * w
    prediction_error = X_b @ w - y
    gradient = (X_b.T @ prediction_error) / (sigma**2) + lmbda * w
    return gradient