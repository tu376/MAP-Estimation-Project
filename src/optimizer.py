import numpy as np

class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01, epochs=1000, tol=1e-6):
        self.lr = learning_rate
        self.epochs = epochs
        self.tol = tol

    def minimize(self, theta_init, grad_func, data, **kwargs):
        theta = np.array(theta_init, dtype=float)
        
        for i in range(self.epochs):
            grad = grad_func(theta, data, **kwargs)
            new_theta = theta - self.lr * grad
            if np.linalg.norm(new_theta - theta) < self.tol:
                break
            theta = new_theta
            
        return theta


def grad_coin_flipping(theta, data, alpha=2, beta=2):
    data = np.array(data)
    k = np.sum(data)   
    n = len(data)     
    theta = np.clip(theta, 1e-10, 1 - 1e-10)
    term1 = (k + alpha - 1) / theta
    term2 = (n - k + beta - 1) / (1 - theta)
    return -(term1 - term2)

def grad_sensor_fusion(theta, data, sigma_likeli=1.0, mu_prior=0, sigma_prior=1.0):
    data = np.array(data)
    grad_likeli = np.sum(theta - data) / (sigma_likeli**2)
    grad_prior = (theta - mu_prior) / (sigma_prior**2)
    
    return grad_likeli + grad_prior

def grad_ridge_regression(w, data, lmbda=0.1, sigma=1.0):
    X, y = data
    X = np.array(X)
    y = np.array(y)
    X_b = np.column_stack([np.ones(len(X)), X])
    prediction_error = X_b @ w - y
    gradient = (X_b.T @ prediction_error) / (sigma**2) + lmbda * w
    return gradient