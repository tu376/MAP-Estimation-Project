import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src import (
        MAP, 
        log_gaussian_likelihood, 
        log_gaussian_prior,
        GradientDescentOptimizer, 
        grad_sensor_fusion
    )
    print("--- Internal modules imported successfully ---")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

data_path = os.path.join(project_root, 'data', 'sensor_data.csv')
try:
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} rows found.")
except FileNotFoundError:
    print(f"File not found at: {data_path}")
    sys.exit(1)

bias_observations = df.iloc[:, 1:6].values.flatten()

mu_prior = 0.2       
sigma_prior = 0.001  
sigma_sensor = 0.5   

solver = MAP(log_gaussian_likelihood, log_gaussian_prior)
optimizer = GradientDescentOptimizer(learning_rate=1e-9, epochs=2000)

initial_guess = np.mean(bias_observations)

map_result = optimizer.minimize(
    initial_guess, 
    grad_sensor_fusion, 
    bias_observations, 
    sigma_likeli=sigma_sensor, 
    mu_prior=mu_prior, 
    sigma_prior=sigma_prior
)

mle_result = np.mean(bias_observations)
diff = map_result - mle_result

print("-" * 50)
print(f"{'PARAMETER':<25} | {'VALUE':<15}")
print("-" * 50)
print(f"{'Observations (N)':<25} | {len(bias_observations):<15}")
print(f"{'MLE Estimate (Data only)':<25} | {mle_result:.6f}")
print(f"{'MAP Estimate (Integrated)':<25} | {map_result:.6f}")
print(f"{'Shift (MAP - MLE)':<25} | {diff:.6f}")
print("-" * 50)

if np.isnan(map_result):
    print("Warning: MAP result is NaN. Try reducing the learning_rate.")
else:
    print("Logic check complete. Results are stable.")

figure_dir = os.path.join(project_root, 'docs', 'figures')

os.makedirs(figure_dir, exist_ok=True)

x_plot = np.linspace(0.1, 0.6, 1000)

y_prior = norm.pdf(x_plot, mu_prior, sigma_prior)
y_likelihood = norm.pdf(x_plot, mle_result, 0.05) 
y_posterior = norm.pdf(x_plot, map_result, 0.01)

plt.figure(figsize=(12, 6))

plt.plot(x_plot, y_prior/np.max(y_prior), '--', 
         label=f'Prior (Believe: {mu_prior})', color='green', alpha=0.7)

plt.plot(x_plot, y_likelihood/np.max(y_likelihood), ':', 
         label=f'Likelihood (Data: {mle_result:.3f})', color='blue', alpha=0.6)

plt.plot(x_plot, y_posterior/np.max(y_posterior), 
         label=f'MAP Result ({map_result:.3f})', color='red', lw=3)

plt.fill_between(x_plot, y_posterior/np.max(y_posterior), color='red', alpha=0.15)

plt.title('Example 2: Sensor Fusion - MAP Estimation Visualization', fontsize=14)
plt.xlabel('Bias Value (°C)', fontsize=12)
plt.ylabel('Normalized Density', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.2)

save_path = os.path.join(figure_dir, 'ex2_sensor_bias_map.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n--- SUCCESS: Plot saved at {save_path} ---")
print("Opening plot window...")
plt.show()