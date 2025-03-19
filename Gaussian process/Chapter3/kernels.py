"""
このファイルはカーネル関数を計算, 描画します

実行する際には
python kernel.py dir_name graph_name kernel_name
を指定してください

例: python kernel.py var_kernel gaussian.png gaussian

kernel_nameは以下のものが指定できます
[gaussian, linear, exp, period, matern3, matern5]

また,kernelの定義に合わせてK = get_kernel()の中身を適宜修正してください

共分散行列を可視化したい場合, Nを非常に小さい値にし, main関数の中のplot_covのコメントアウトを外してください
"""

# Information
# File: kernel.py
# Author: Fukuda Akiya
# Date: 2025-03-19
# Description: plotting various kernel function

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal
import math
import seaborn as sns
import random

random.seed(123)

# param
xmax = 5
xmin = -5
ymax = 5
ymin = -5
N = 100
theta1 = 1
theta2 = 1


# ガウスカーネルの定義
def gaussian_kernel(x1, x2, theta1, theta2):
	sqdist = np.subtract.outer(x1, x2) ** 2
	return theta1 * np.exp(-abs(sqdist)/theta2)

# 線形カーネル
def linear_kernel(x1, x2):
	return np.outer(x1, x2)

# 指数カーネル
def exp_kernel(x1, x2, theta1):
	sqdist = np.subtract.outer(x1, x2)
	kernel_matrix = np.exp(-abs(sqdist) / theta)
	return kernel_matrix

# 周期カーネル
def period_kernel(x1, x2, theta1, theta2):
	sqdist = np.subtract.outer(x1, x2)
	kernel_matrix = np.exp(theta1 * np.cos(abs(sqdist)/theta2))
	return kernel_matrix

# marternカーネル
def matern_kernel3(x1, x2, theta1):
	r = np.subtract.outer(x1, x2)
	kernel_matrix = (1 + ((np.sqrt(3) * abs(r))/ theta)) * np.exp(-((np.sqrt(3) * abs(r)) / theta))
	return kernel_matrix

def matern_kernel5(x1, x2, theta1):
	r = abs(np.subtract.outer(x1, x2))
	kernel_matrix = (1 + (np.sqrt(5) * r / theta) + (5 * r ** 2 / 3 * theta ** 2 )) * np.exp(-np.sqrt(5) * r /theta)
	return kernel_matrix

# カーネル選択関数
def get_kernel(kernel_type, x1, x2, *params):
    if kernel_type == "gaussian":
        return gaussian_kernel(x1, x2, *params)
    elif kernel_type == "linear":
        return linear_kernel(x1, x2)
    elif kernel_type == "exp":
        return exp_kernel(x1, x2, *params)
    elif kernel_type == "period":
        return period_kernel(x1, x2, *params)
    elif kernel_type == "matern3":
        return matern_kernel3(x1, x2, *params)
    elif kernel_type == "matern5":
    	return matern_kernel5(x1, x2, *params)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def plot_kernel(f, x, dir_n, file_n):
	for i in range(len(f)):
		    plt.plot(x, f[i], label=f"Sample {i+1}")

	plt.title("Samples from a Gaussian Process")
	plt.xlabel("x")
	plt.ylabel("f(x)")
	plt.xlim(xmin,xmax)
	plt.ylim(ymin,ymax)
	plt.legend()
	plt.savefig(f"{dir_n}/{file_n}")
	plt.clf()
	plt.close()
	
def plot_cov(K, dir_n, file_n):
	sns.heatmap(K, square=True, cmap='Blues')
	plt.savefig(f"{dir_n}/{file_n}")

def usage ():
    print('Usage: python script.py <dir> <file_name> <kernel_type>')
    sys.exit (0)

def main():
    if len(sys.argv) < 4:
        usage()
        sys.exit(1)

    # コマンドライン引数の読み込み
    dir_n = sys.argv[1]  # ディレクトリ
    file_n = sys.argv[2]  # ファイル名
    kernel_name = sys.argv[3]  # カーネル名

    # 入力点の生成
    X1 = np.linspace(xmin, xmax, N)

    # 共分散行列 K を計算
    K = get_kernel(kernel_name, X1, X1, theta1, theta2)

    # ガウス過程からサンプル（sizeは関数の数）
    f = np.random.multivariate_normal(mean=np.zeros(N), cov=K, size=4)

    # y のグラフをプロット
    plot_kernel(f, X1, dir_n, file_n)

    # 共分散行列のヒートマップ
    #plot_cov(K, dir_n, file_n)

# メイン関数を実行
if __name__ == "__main__":
    main()
