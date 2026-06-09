import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import MAP, log_linear_likelihood, log_linear_prior
from src import GradientDescentOptimizer, grad_ridge_regression

from src.utils import set_seed, load_csv, train_test_split, add_bias_column

set_seed(42)

data_path = os.path.join(project_root, 'data', 'cleaned_insurance.csv')
df = load_csv(data_path)

feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker',
                'region_northwest', 'region_southeast', 'region_southwest']
X = df[feature_cols].values
y = df['charges_log'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sigma_likeli = 0.5     
sigma_prior = 0.4      
lambda_prior = 1.0 / (sigma_prior**2)
learning_rate = 1e-5
epochs = 5000

initial_w = np.zeros(X_train.shape[1] + 1)  
optimizer = GradientDescentOptimizer(learning_rate=learning_rate, epochs=epochs)

data_tuple = (X_train, y_train)
map_weights = optimizer.minimize(
    theta_init=initial_w,
    grad_func=grad_ridge_regression,
    data=data_tuple,
    lmbda=lambda_prior,
    sigma=sigma_likeli
)

initial_w_mle = np.zeros(X_train.shape[1] + 1)

def grad_ols(theta, data, lmbda=0, sigma=sigma_likeli):
    X, y = data 
    X_b = add_bias_column(X, prepend=True) 
    residuals = X_b @ theta - y
    grad = (X_b.T @ residuals) / (sigma**2)
    return grad

optimizer_mle = GradientDescentOptimizer(learning_rate=learning_rate, epochs=epochs)

mle_weights = optimizer_mle.minimize(
    theta_init=initial_w_mle,
    grad_func=grad_ols,
    data=data_tuple,
    lmbda=0,  
    sigma=sigma_likeli
)

prior_mean = np.zeros_like(mle_weights)  

param_names = ['Intercept'] + feature_cols

x = np.arange(len(param_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))

ax.scatter(x, prior_mean, color='gray', s=200, marker='s', 
           label='Prior Mean (μ=0)', zorder=5, edgecolors='black', linewidth=1.5)

rects2 = ax.bar(x - width/2, mle_weights, width, 
                label='MLE (λ=0)', 
                color='steelblue', alpha=0.7, edgecolor='black')

rects3 = ax.bar(x + width/2, map_weights, width, 
                label=f'MAP (Ridge, λ={lambda_prior:.2f})', 
                color='salmon', alpha=0.7, edgecolor='black')

ax.set_ylabel('Coefficient Value', fontsize=12)
ax.set_title(f'Comparison: Prior Mean (dots) vs MLE vs MAP\n(Gaussian Prior with Mean=0, Std={sigma_prior})', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
ax.legend(loc='best', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

def add_values(bars, color='black'):
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.01:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45, color=color)

add_values(rects2, color='steelblue')
add_values(rects3, color='salmon')

plt.tight_layout()
plt.show()

if len(param_names) >= 3:
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    ax2.scatter(prior_mean[1], prior_mean[2], 
                color='gray', s=200, marker='*', label='Prior Mean (0,0)', 
                zorder=5, edgecolors='black', linewidth=1)
    
    ax2.scatter(mle_weights[1], mle_weights[2], 
                color='steelblue', s=150, marker='o', label=f'MLE', 
                zorder=4, edgecolors='black', linewidth=1)
    
    ax2.scatter(map_weights[1], map_weights[2], 
                color='salmon', s=150, marker='^', label=f'MAP', 
                zorder=4, edgecolors='black', linewidth=1)
    
    ax2.plot([prior_mean[1], mle_weights[1]], [prior_mean[2], mle_weights[2]], 
             'b--', alpha=0.5, label='MLE direction', linewidth=1)
    ax2.plot([prior_mean[1], map_weights[1]], [prior_mean[2], map_weights[2]], 
             'r--', alpha=0.5, label='MAP direction', linewidth=1)
    
    ax2.annotate(f'MLE: ({mle_weights[1]:.2f}, {mle_weights[2]:.2f})', 
                 xy=(mle_weights[1], mle_weights[2]), xytext=(10, 10),
                 textcoords='offset points', fontsize=9, color='steelblue')
    ax2.annotate(f'MAP: ({map_weights[1]:.2f}, {map_weights[2]:.2f})', 
                 xy=(map_weights[1], map_weights[2]), xytext=(10, -10),
                 textcoords='offset points', fontsize=9, color='salmon')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel(f'{param_names[1]} coefficient', fontsize=12)
    ax2.set_ylabel(f'{param_names[2]} coefficient', fontsize=12)
    ax2.set_title(f'2D Visualization: Shrinkage effect from MLE to MAP\n(Prior centered at (0,0), σ_prior={sigma_prior})', 
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

'''
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import MAP, log_linear_likelihood, log_linear_prior
from src import GradientDescentOptimizer, grad_ridge_regression

data_path = os.path.join(project_root, 'data', 'cleaned_insurance.csv')
df = pd.read_csv(data_path)

feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker',
                'region_northwest', 'region_southeast', 'region_southwest']
X = df[feature_cols].values
y = df['charges_log'].values

np.random.seed(42)

shuffled_indices = np.random.permutation(len(X))
test_size = 0.2
test_set_size = int(len(X) * test_size)

test_indices = shuffled_indices[:test_set_size]
train_indices = shuffled_indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

sigma_likeli = 0.5      
sigma_prior = 0.4      
lambda_prior = 1.0 / (sigma_prior**2)
learning_rate = 1e-5
epochs = 5000

initial_w = np.zeros(X_train.shape[1] + 1)   
optimizer = GradientDescentOptimizer(learning_rate=learning_rate, epochs=epochs)

data_tuple = (X_train, y_train)
map_weights = optimizer.minimize(
    theta_init=initial_w,
    grad_func=grad_ridge_regression,
    data=data_tuple,
    lmbda=lambda_prior,
    sigma=sigma_likeli
)

initial_w_mle = np.zeros(X_train.shape[1] + 1)

def grad_ols(theta, data, lmbda=0, sigma=sigma_likeli):
    X, y = data 
    X_b = np.column_stack([np.ones(len(X)), X])
    residuals = X_b @ theta - y
    grad = (X_b.T @ residuals) / (sigma**2)
    return grad

optimizer_mle = GradientDescentOptimizer(learning_rate=learning_rate, epochs=epochs)

mle_weights = optimizer_mle.minimize(
    theta_init=initial_w_mle,
    grad_func=grad_ols,
    data=data_tuple,
    lmbda=0,  
    sigma=sigma_likeli
)

prior_mean = np.zeros_like(mle_weights)  

param_names = ['Intercept'] + feature_cols

x = np.arange(len(param_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))

ax.scatter(x, prior_mean, color='gray', s=200, marker='s', 
           label='Prior Mean (μ=0)', zorder=5, edgecolors='black', linewidth=1.5)

rects2 = ax.bar(x - width/2, mle_weights, width, 
                label='MLE (λ=0)', 
                color='steelblue', alpha=0.7, edgecolor='black')

rects3 = ax.bar(x + width/2, map_weights, width, 
                label=f'MAP (Ridge, λ={lambda_prior:.2f})', 
                color='salmon', alpha=0.7, edgecolor='black')

ax.set_ylabel('Coefficient Value', fontsize=12)
ax.set_title(f'Comparison: Prior Mean (dots) vs MLE vs MAP\n(Gaussian Prior with Mean=0, Std={sigma_prior})', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=10)
ax.legend(loc='best', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

def add_values(bars, color='black'):
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.01:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45, color=color)

plt.tight_layout()
plt.show()

if len(param_names) >= 3:
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    ax2.scatter(prior_mean[1], prior_mean[2], 
                color='gray', s=200, marker='*', label='Prior Mean (0,0)', 
                zorder=5, edgecolors='black', linewidth=1)
    
    ax2.scatter(mle_weights[1], mle_weights[2], 
                color='steelblue', s=150, marker='o', label=f'MLE', 
                zorder=4, edgecolors='black', linewidth=1)
    
    ax2.scatter(map_weights[1], map_weights[2], 
                color='salmon', s=150, marker='^', label=f'MAP', 
                zorder=4, edgecolors='black', linewidth=1)
    
    ax2.plot([prior_mean[1], mle_weights[1]], [prior_mean[2], mle_weights[2]], 
             'b--', alpha=0.5, label='MLE direction', linewidth=1)
    ax2.plot([prior_mean[1], map_weights[1]], [prior_mean[2], map_weights[2]], 
             'r--', alpha=0.5, label='MAP direction', linewidth=1)
    
    ax2.annotate(f'MLE: ({mle_weights[1]:.2f}, {mle_weights[2]:.2f})', 
                 xy=(mle_weights[1], mle_weights[2]), xytext=(10, 10),
                 textcoords='offset points', fontsize=9, color='steelblue')
    ax2.annotate(f'MAP: ({map_weights[1]:.2f}, {map_weights[2]:.2f})', 
                 xy=(map_weights[1], map_weights[2]), xytext=(10, -10),
                 textcoords='offset points', fontsize=9, color='salmon')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel(f'{param_names[1]} coefficient', fontsize=12)
    ax2.set_ylabel(f'{param_names[2]} coefficient', fontsize=12)
    ax2.set_title(f'2D Visualization: Shrinkage effect from MLE to MAP\n(Prior centered at (0,0), σ_prior={sigma_prior})', 
                  fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
'''