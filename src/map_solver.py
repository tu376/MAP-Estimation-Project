import numpy as np

class MAP:
    def __init__(self, log_likelihood, log_prior):
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior

    def log_posterior(self, theta, data):
        return self.log_likelihood(theta, data) + self.log_prior(theta)
    
    def posterior(self, theta, data):
        return np.exp(self.log_posterior(theta, data))

# Bernoulli for Coin Flipping

def log_bernoulli_likelihood(theta , data):
    """
    theta: The probability of result 1
    data: An array of (0, 1)
    """
    data = np.array(data) # Array of (0, 1)

    if theta <= 0 or theta >= 1:
        return -np.inf
    
    m = np.sum(data) # Number of 1
    n = len(data) - m # Number of 0

    return m * np.log(theta) + n * np.log(1 - theta)

def log_bernoulli_prior(theta, alpha=2, beta=2): # Beta Distribution
    """
    theta: The probability of result 1
    alpha: Number of result 1
    beta: Number of result 0
    """

    if theta <= 0 or theta >= 1:
        return -np.inf
    
    return (alpha - 1) * np.log(theta) + (beta - 1) * np.log(1 - theta)

# Gaussian for Sensor Fusion

def log_gaussian_likelihood(theta, data, sigma=1):
    """
    theta: Mean
    sigma: Standard Deviation (Equals 1 cause it does not effect the final answer)
    data: An array of values
    """

    data = np.array(data)

    return - sum((data - theta)**2) / (2 * sigma**2)

def log_gaussian_prior(theta, mu=0, sigma=1):
    """
    theta: Mean
    sigma: Standard Deviation
    mu: Mean (Equals 0 for easy caculation)
    """

    return - np.sum((theta - mu)**2) / (2 * sigma**2)

# Linear Regression

def log_linear_likelihood(theta, data, sigma=1.0):
    """
    theta: Array [w0, w1, ..., wn] — intercept + weights
    data : Tuple (X, y)
             X: 2D array, shape (n_samples, n_features)
             y: 1D array, shape (n_samples,)
    sigma: Noise std (assumed known)
    """
    X, y = data
    X = np.array(X)
    y = np.array(y)

    # Thêm cột 1 vào X để tính bias (w0)
    X_b = np.column_stack([np.ones(len(X)), X])

    y_pred = X_b @ theta          # ŷ = Xθ
    residuals = y - y_pred        # e = y - ŷ

    return -np.sum(residuals**2) / (2 * sigma**2)


def log_linear_prior(theta, sigma=1.0):
    """
    Gaussian prior (Ridge): w ~ N(0, sigma²I)
    theta: Array [w0, w1, ..., wn]
    """
    return -np.sum(theta**2) / (2 * sigma**2)