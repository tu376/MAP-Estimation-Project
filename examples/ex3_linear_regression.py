import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Đường dẫn tới module src (tuỳ chỉnh theo cây thư mục của bạn)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import MAP, log_linear_likelihood, log_linear_prior
from src import GradientDescentOptimizer, grad_ridge_regression

# ------------------------------
# 1. Đọc dữ liệu
# ------------------------------
data_path = os.path.join(project_root, 'data', 'cleaned_insurance.csv')
df = pd.read_csv(data_path)

feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker',
                'region_northwest', 'region_southeast', 'region_southwest']
X = df[feature_cols].values
y = df['charges_log'].values

# Thêm cột 1 cho intercept (sẽ được xử lý bên trong grad_ridge_regression, nhưng cần đảm bảo)
# Hàm grad_ridge_regression tự thêm cột 1 nếu X chưa có intercept.
# Không cần thêm thủ công.

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# 2. Hyperparameters
# ------------------------------
sigma_likeli = 0.5      # nhiễu của likelihood
sigma_prior = 1.0       # prior std (điều khiển độ co)
lambda_prior = 1.0 / (sigma_prior**2)
learning_rate = 1e-5
epochs = 5000

# ------------------------------
# 3. MAP estimation
# ------------------------------
initial_w = np.zeros(X_train.shape[1] + 1)   # +1 intercept
optimizer = GradientDescentOptimizer(learning_rate=learning_rate, epochs=epochs)

data_tuple = (X_train, y_train)
best_w = optimizer.minimize(
    theta_init=initial_w,
    grad_func=grad_ridge_regression,
    data=data_tuple,
    lmbda=lambda_prior,
    sigma=sigma_likeli
)

# ------------------------------
# 4. In kết quả
# ------------------------------
print("\n=== MAP Estimates (Ridge Regression) ===")
print(f"{'Intercept':<20}: {best_w[0]:.6f}")
for i, col in enumerate(feature_cols):
    print(f"{col:<20}: {best_w[i+1]:.6f}")

# ------------------------------
# 5. (Tuỳ chọn) Đánh giá trên test set
# ------------------------------
X_test_b = np.column_stack([np.ones(len(X_test)), X_test])
y_pred = X_test_b @ best_w
mse = np.mean((y_test - y_pred)**2)
print(f"\nMSE trên test set: {mse:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")