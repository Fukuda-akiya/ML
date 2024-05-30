using Random
using Plots
using Distributions
using StatsBase
using Printf

# 乱数を固定値に設定
Random.seed!(123)

# mu=0.9でベルヌーイ分布を計算するように設定
bernoulli_dist = Bernoulli(0.9)

# サンプル数
num_samples = 20

# 乱数でベルヌーイ分布を計算
samples = rand(bernoulli_dist, num_samples)

# 値の頻度を計算
counts = countmap(samples)

# グラフの描画
#bar(["0", "1"], [counts[0], counts[1]], xlabel="Sample", ylabel="Frequency", title="Frequency of Bernoulli Distribution Samples", yticks=0:1:num_samples)
#savefig("bernoulli_samples0.9.png")

# Entropyの計算
N = 10
delta = 1/N
H = zeros(N)
mu = zeros(N)
mu_hat = zeros(N)


for i in 1:N
	mu[i] = delta * (i-1)
	mu_hat[i] = delta * (i-1)
	H[i] = ( - mu[i] * log(mu[i]) ) - ( ( 1 - mu[i] ) * log( 1 - mu[i] ) )
end

# muとHの関係性を描画
#plot(mu, H, xlims=(0, 1.0), ylims=(0, 0.7), xlabel="mu", ylabel="Entropy", legend=false, linewidth=1)
#savefig("bernoulli_entropy.png")

# KLダイバージェンスの計算
function KL_cal(mu,mu_hat)
	KL = zeros(N, N)
	for i in 1:N
		for j in 1:N
			KL[i, j] = ( mu_hat[i] * log(mu_hat[i]) ) + ( (1 - mu_hat[i]) * log(1 - mu_hat[i]) ) - mu_hat[i] * log(mu[j]) - (1 - mu_hat[i]) * log(1 - mu[j])
		end
	end
	return KL
end


result = KL_cal(mu,mu_hat)

fp = open("KL_div_bern.dat","w")

@printf(fp,"mu_hat\tmu\tKL\n")
for i in 1:N
	for j in 1:N
		@printf(fp, "%.06e\t%.06e\t%.06e\n",mu_hat[i], mu[j], result[i,j])
	end 
end 
close(fp)
