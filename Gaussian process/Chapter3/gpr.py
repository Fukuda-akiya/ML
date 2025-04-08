"""
このファイルはガウス過程回帰を実行します.

実行する際には
python grp.py traning_data output.png
を指定してください

例: python grp.py gpr_data.dat gpr.png 


なお, ハイパーパラメータチューニングはしていません.
また, 訓練データは(x,y)としています.
"""

# Information
# File: gpr.py
# Author: Fukuda Akiya
# Date: 2025-04-08
# Description: perform regression based on gauss process

import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# plot parameters
N	= 100
xmin	= -1
xmax	= 3.5
ymin	= -1
ymax	= 3.5

# kernel parameters
eta 	= 0.1
tau 	= 1
sigma	= 1

def kgauss(tau, sigma):
	return lambda x,y: tau * np.exp(-(x - y)**2 / (2 * sigma**2))

def kernel_matrix(xx, kernel):
	N = len(xx)
	K = np.zeros((N,N))
	for i,xi in enumerate(xx):
		for j,xj in enumerate(xx):
			K[i,j] = np.array(kernel(xi, xj))

	noise = eta * np.eye(N)
	return K + noise

def K_star(x, xtrain, kernel):
	k_star = np.array([kernel(x, xi) for xi in xtrain])
	return k_star

def gpr(xx, xtrain, ytrain, kernel):
	# 予測分布 y の配列
	ypr = []
	
	# 予測分布 y の分散の配列
	spr = []
	
	# 訓練データのカーネル行列
	K = kernel_matrix(xtrain, kernel)
	
	# 逆行列の準備
	Kinv = inv(K)

	# k*の計算
	for x in xx:
		# 訓練データと予測x*の共分散行列を計算
		k_star = K_star(x, xtrain, kernel)
		
		# y の推論
		y_pred = k_star.T @ Kinv @ ytrain
		ypr.append(y_pred)
		
		# k_**の準備
		s = kernel(x, x) + eta
		
		# s の計算
		s_pred = s - k_star.T @ Kinv @ k_star
		spr.append(s_pred)
	
	return ypr, spr

def plot(xtrain, ytrain, xx, ypr, spr, file_name):
	plt.plot(xtrain,ytrain,'bx',markersize=16)
	plt.plot(xx, ypr, 'b-')
	plt.fill_between(xx, ypr - 2*np.sqrt(spr), ypr + 2*np.sqrt(spr), color='#ccccff')
	plt.xlim(xmin,xmax)
	plt.ylim(ymin,ymax)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig(file_name, bbox_inches='tight')

def usage():
	print('Usage: python grp_regression.py <input> <output>')
	sys.exit(0)

def main():
	if len(sys.argv) < 3:
		usage()
		sys.exit(1)

	# コマンドライン引数の読み込み
	train = np.loadtxt(sys.argv[1], dtype=float)
	file_name = sys.argv[2]
	
	# 訓練データ
	xtrain = train[:,0]
	ytrain = train[:,1]
	
	# カーネル関数の準備
	kernel = kgauss(tau, sigma)
	
	# 回帰するグリッドの準備
	xx = np.linspace(xmin, xmax, N)
	
	# ガウス過程回帰
	ypr, spr = gpr(xx, xtrain, ytrain, kernel)

	plot(xtrain,ytrain,xx,ypr,spr,file_name)

if __name__ == "__main__":
	main()
