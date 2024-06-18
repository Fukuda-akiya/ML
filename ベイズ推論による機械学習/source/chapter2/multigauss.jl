using Distributions
using Plots

# 2次元ガウス分布のパラメータ
D = 2  # 次元数
mu = [0.0, 0.0]  # 平均ベクトル
sigma = [4.0 sqrt(3) ;
	 sqrt(3) 2.0]  # 共分散行列

# D次元のガウス分布のインスタンス生成
dist = MvNormal(mu, sigma)

# 変数
x = -4:0.01:4
y = -4:0.01:4

# 確率密度関数
z = [pdf(dist, [i, j]) for i in x, j in y]

# プロット
plot(title="2D Gaussian Distributions", xlabel="x", ylabel="y",zlim=(0,0.08))
plot!(x, y, z, st = :surface, color=:rainbow)
savefig("multi_gauss_sample03.png")
