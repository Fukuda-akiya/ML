using Random
using Plots
using Distributions
using StatsBase

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
N = 1024
delta = 1/N
H = zeros(N)
mu = zeros(N)

for i in 1:N
	mu[i] = delta * i
	H[i] = ( - mu[i] * log(mu[i]) ) - ( ( 1 - mu[i] ) * log( 1 - mu[i] ) )
end

# muとHの関係性を描画
plot(mu, H, xlims=(0, 1.0), ylims=(0, 0.7), xlabel="mu", ylabel="Entropy", legend=false, linewidth=1)
savefig("bernoulli_entropy.png")
