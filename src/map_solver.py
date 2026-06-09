import numpy as np

class MAP:
    def __init__(self, log_likelihood, log_prior):
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior

    def log_posterior(self, theta, data):
        return self.log_likelihood(theta, data) + self.log_prior(theta)
    
    def posterior(self, theta, data):
        return np.exp(self.log_posterior(theta, data))

def log_bernoulli_likelihood(theta , data):
    data = np.array(data)

    if theta <= 0 or theta >= 1:
        return -np.inf
    
    m = np.sum(data) 
    n = len(data) - m 

    return m * np.log(theta) + n * np.log(1 - theta)

def log_bernoulli_prior(theta, alpha=2, beta=2):
    if theta <= 0 or theta >= 1:
        return -np.inf
    
    return (alpha - 1) * np.log(theta) + (beta - 1) * np.log(1 - theta)

def log_gaussian_likelihood(theta, data, sigma=1):
    data = np.array(data)

    return - sum((data - theta)**2) / (2 * sigma**2)

def log_gaussian_prior(theta, mu=0, sigma=1):
    return - np.sum((theta - mu)**2) / (2 * sigma**2)

def log_linear_likelihood(theta, data, sigma=1.0):
    X, y = data
    X = np.array(X)
    y = np.array(y)
    X_b = np.column_stack([np.ones(len(X)), X])

    y_pred = X_b @ theta        
    residuals = y - y_pred        

    return -np.sum(residuals**2) / (2 * sigma**2)

def log_linear_prior(theta, sigma=1.0):
    return -np.sum(theta**2) / (2 * sigma**2)