using Distributions
using Plots
using Printf

# ベータ分布のパラメータ
a = [0.5, 0.6, 1.0, 10.0, 10.0]
b = [0.5, 0.8, 1.0, 40.0, 5.0]

# ベータ分布のインスタンスの作成
beta_dist = [Beta(a[i],b[i]) for i in 1:length(a)]

# 確率密度関数の作成
mu = 0:0.001:1.0
for i in 1:length(beta_dist)
	density = pdf.(beta_dist[i], mu)
	plot!(mu, density, label="α=$(a[i]), β=$(b[i])", xlims=(0,1.0), ylims=(0,8.0))
end
savefig("beta_dist.png")
