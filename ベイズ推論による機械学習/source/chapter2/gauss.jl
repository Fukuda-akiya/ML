using Distributions
using Plots
using Printf

# パラメータ
mu = [0.0, 2.0, -6.0]
sigma = [1.0, 2.0, 0.3]

# ガウス分布のインスタンス
dist = [Normal(mu[i], sigma[i]) for i in 1:length(mu)]

# xの範囲
x = -10.0:0.01:10.0

#=
# プロットの準備
plot(title="Gaussian Distributions", xlabel="x", ylabel="Density")
plot(xlim=(-10.0,10.0),ylim=(0,1.4))

# 各ガウス分布に対してPDFを計算してプロット
for i in 1:length(dist)
	y = pdf.(dist[i], x)
	plot!(x, y, label="μ=$(mu[i]), σ=$(sigma[i])")
end

savefig("gauss_dist.png")
=#




# KLダイバージェンス
function KL_cal(mu, mu_hat, sigma, sigma_hat)
	 KL = 0.5 * ( ( ( (mu - mu_hat)^2 + sigma_hat^2 ) * sigma^-2)
			 	    + log(sigma^2 * sigma_hat^-2) 
				    - 1 )
	 return KL

end

# パラメータ	
mu = 0.0
mu_hat = 2.0
sigma = 2.0
sigma_hat = 1.0

# KLダイバージェンスの計算
result1 = KL_cal(mu, mu_hat, sigma, sigma_hat)
result2 = KL_cal(mu_hat, mu, sigma_hat, sigma)
println(result1)
println(result2)

# ガウス分布のインスタンス
dist1 = Normal(mu, sigma)
dist2 = Normal(mu_hat, sigma_hat)

x = -8:0.01:8
y1 = pdf(dist1, x)
y2 = pdf(dist2, x)

plot(title="Gaussian Distributions", xlabel="x", ylabel="Density")
plot!(x, y1, label="mu=$(mu), sigma=$(sigma)")
plot!(x, y2, label="mu=$(mu_hat), sigma=$(sigma_hat)")
savefig("Gauss_KL03.png")
