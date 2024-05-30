using Random
using Distributions
using Plots

M_trial = 10  # 試行回数
mu = 0.2  # パラメータ

binomial_dist = Binomial(M_trial, mu)

x = 0:M_trial  # 正解数
y = pdf(binomial_dist, x)  # 確率質量関数の計算

bar(x, y, ylims=(0,0.35), xticks=0:1:20,xlabel="Number of successes", ylabel="Probability", title="Binomial Distribution (n=10, p=0.2)", legend=false)
savefig("binomial_M10_mu02.png")
