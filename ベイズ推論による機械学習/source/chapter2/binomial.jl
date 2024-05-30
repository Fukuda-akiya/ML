using Random
using Distributions
using Plots

M_trial = 10
mu = 0.2

binomial_dist = Binomial(M_trial, mu)

x = 0:M_trial
y = pdf(binomial_dist, x)

bar(x, y, ylims=(0,0.35), xticks=0:1:20,xlabel="Number of successes", ylabel="Probability", title="Binomial Distribution (n=10, p=0.2)", legend=false)
savefig("binomial_M10_mu02.png")
