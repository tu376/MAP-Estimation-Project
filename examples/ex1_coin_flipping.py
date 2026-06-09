import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src import (
        MAP,
        log_bernoulli_likelihood,
        log_bernoulli_prior,
        GradientDescentOptimizer,
        grad_coin_flipping,
    )
    print("--- Internal modules imported successfully ---")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def mle_bernoulli(data):
    data = np.array(data)
    return np.mean(data)


def map_closed_form_beta_bernoulli(data, alpha=2, beta=2):
    data = np.array(data)
    k = np.sum(data)
    n = len(data)
    return (k + alpha - 1) / (n + alpha + beta - 2)


def plot_comparison(data, theta_mle, theta_map, alpha, beta_prior):
    data = np.array(data)
    k = int(np.sum(data))
    n = len(data)

    x = np.linspace(1e-4, 1 - 1e-4, 1000)

    y_prior = beta.pdf(x, alpha, beta_prior)
    y_posterior = beta.pdf(x, alpha + k, beta_prior + n - k)

    y_likelihood_like = beta.pdf(x, k + 1, n - k + 1)

    y_prior = y_prior / np.max(y_prior)
    y_posterior = y_posterior / np.max(y_posterior)
    y_likelihood_like = y_likelihood_like / np.max(y_likelihood_like)

    plt.figure(figsize=(11, 6))
    plt.plot(x, y_prior, '--', color='green', lw=2, label=f'Prior Beta({alpha}, {beta_prior})')
    plt.plot(x, y_likelihood_like, ':', color='blue', lw=2, label='Likelihood shape (scaled)')
    plt.plot(x, y_posterior, color='red', lw=3, label='Posterior Beta(alpha+k, beta+n-k)')

    plt.axvline(theta_mle, color='blue', linestyle='--', alpha=0.7, label=f'MLE = {theta_mle:.3f}')
    plt.axvline(theta_map, color='red', linestyle='-.', alpha=0.8, label=f'MAP = {theta_map:.3f}')

    plt.title('Example 1: Coin Flipping (Bernoulli-Beta) - MLE vs MAP', fontsize=13)
    plt.xlabel('Theta (probability of heads)', fontsize=11)
    plt.ylabel('Normalized density', fontsize=11)
    plt.grid(alpha=0.2)
    plt.legend()

    output_dir = os.path.join(project_root, 'docs','figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ex1_coin_flipping_mle_vs_map.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n--- SUCCESS: Plot saved at {output_path} ---")
    plt.show()


def compare_methods(data, alpha=2, beta=2):
    theta_mle = mle_bernoulli(data)
    solver = MAP(log_bernoulli_likelihood, lambda t: log_bernoulli_prior(t, alpha=alpha, beta=beta))
    optimizer = GradientDescentOptimizer(learning_rate=0.01, epochs=5000, tol=1e-10)

    initial_guess = np.clip(theta_mle, 1e-3, 1 - 1e-3)

    theta_map_opt = optimizer.minimize(
        theta_init=initial_guess,
        grad_func=grad_coin_flipping,
        data=data,
        alpha=alpha,
        beta=beta,
    )
    theta_map_opt = float(np.clip(theta_map_opt, 1e-10, 1 - 1e-10))
    theta_map_closed = map_closed_form_beta_bernoulli(data, alpha=alpha, beta=beta)

    log_post_mle = solver.log_posterior(theta_mle, data)
    log_post_map = solver.log_posterior(theta_map_opt, data)

    print("\n" + "=" * 72)
    print("EXAMPLE 1: COIN FLIPPING (Bernoulli-Beta)")
    print("=" * 72)
    print(f"Observations (n): {len(data)}")
    print(f"Heads (k): {int(np.sum(data))}")
    print(f"Tails (n-k): {int(len(data) - np.sum(data))}")
    print(f"Beta prior: alpha={alpha}, beta={beta}")

    print("\nEstimation results")
    print("-" * 72)
    print(f"MLE (closed-form):               {theta_mle:.6f}")
    print(f"MAP (optimizer):                 {theta_map_opt:.6f}")
    print(f"MAP (closed-form, validation):   {theta_map_closed:.6f}")
    print(f"|MAP_opt - MAP_closed|:          {abs(theta_map_opt - theta_map_closed):.6e}")
    print("-" * 72)
    print(f"log-posterior at MLE:            {log_post_mle:.6f}")
    print(f"log-posterior at MAP:            {log_post_map:.6f}")

    print("\nDifferences from other methods (MLE vs MAP)")
    print("-" * 72)
    print("MLE - Pros:")
    print("  + Very simple; often closed-form and fast.")
    print("  + Uses only observed data, easy to interpret.")
    print("MLE - Cons:")
    print("  - Can be unstable with small samples.")
    print("  - No mechanism to include prior domain knowledge.")

    print("MAP - Pros:")
    print("  + Combines data evidence with prior belief.")
    print("  + More robust in low-data settings (regularization effect).")
    print("MAP - Cons:")
    print("  - Requires selecting a prior (can introduce bias if poorly chosen).")
    print("  - May require numerical optimization in complex models.")

    print("\nInterpretation for this run")
    print("-" * 72)
    shift = theta_map_opt - theta_mle
    print(f"MAP - MLE shift: {shift:+.6f}")
    if shift > 0:
        print("Posterior moved upward due to prior/data combination.")
    elif shift < 0:
        print("Posterior moved downward due to prior/data combination.")
    else:
        print("No shift observed (rare unless prior/data align exactly).")

    plot_comparison(data, theta_mle, theta_map_opt, alpha, beta)


if __name__ == "__main__":
    flips = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    alpha_prior = 2
    beta_prior = 2
    compare_methods(flips, alpha=alpha_prior, beta=beta_prior)