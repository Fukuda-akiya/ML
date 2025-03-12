import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import math

# ガウスカーネルの定義
def gaussian_kernel(x1, x2, theta1, theta2):
	sqdist = np.subtract.outer(x1, x2) ** 2
	return theta1 * np.exp(-abs(sqdist)/theta2)

# 線形カーネル
def linear_kernel(x1, x2):
	return np.outer(x1, x2)

# 指数カーネル
def exp_kernel(x1, x2, theta):
	sqdist = np.subtract.outer(x1, x2)
	kernel_matrix = np.exp(-abs(sqdist) / theta)
	return kernel_matrix

# 周期カーネル
def period_kernel(x1, x2, theta1, theta2):
	sqdist = np.subtract.outer(x1, x2)
	kernel_matrix = np.exp(theta1 * np.cos(abs(sqdist)/theta2))
	return kernel_matrix

# 入力点の生成
X = np.linspace(-4,4,100)

# 共分散行列 K を計算
#K = gaussian_kernel(X, X, 1.0, 1.0)
#K = linear_kernel(X, X)
#K = exp_kernel(X, X, 1.0)
K = period_kernel(X, X, 0.5, 0.5)

# ガウス過程からサンプル（sizeは関数の数）
f = np.random.multivariate_normal(mean=np.zeros(len(X)), cov=K, size=4)

# 描画
plt.figure(figsize=(8, 4))
for i in range(len(f)):
	    plt.plot(X, f[i], label=f"Sample {i+1}")

plt.title("Samples from a Gaussian Process with linear Kernel")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim(-4,4)
plt.legend()
plt.savefig("var_kernel/period.png")
