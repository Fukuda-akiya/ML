using Distributions
using Plots

# ディリクレ分布のパラメータ
alpha = [0.5, 0.5, 0.5]

# ディリクレ分布のインスタンスを生成
dist = Dirichlet(alpha)

# ディリクレ分布の確率密度関数
function dirichlet_pdf(dist, x)
    return pdf(dist, x)
end

# 2次元平面上の点(x, y)を生成
n_points = 10000
x = range(0, stop=1, length=n_points)
y = range(0, stop=1, length=n_points)


# 確率密度を計算
z = zeros(n_points, n_points)
for i in 1:n_points
	for j in 1:n_points
		if x[i] + y[j] <= 1
			z[i, j] = dirichlet_pdf(dist, [x[i], y[j], 1 - x[i] - y[j]])
		else
			z[i, j] = NaN  # ディリクレ分布の範囲外の点
		end
	end
end

z_limited = clamp.(z, 0, 25)

# 3次元プロットの作成
plot(x, y, z_limited, title="Dirichlet Distribution PDF", xlabel="x", ylabel="y",  st = :surface, zlim=(0,25),clim=(0,100))
savefig("diri_dist_sample2.png")
