using Distributions
using Plots
using Printf
using Random
using StatsBase

# 乱数の固定値に設定
Random.seed!(123)

#####################################
##### モデルの生成（正解モデル）#####
#####################################
# 真のパラメータ
mu_truth = 0.25

# ベルヌーイ分布のインスタンス作成
bernoulli_dist_truth = Bernoulli(mu_truth)

# 確率質量関数（PMF）を計算
x_values = [0, 1]
pmf_values = [pdf(bernoulli_dist_truth, x) for x in x_values]

# PMFの確認
for (x, pmf) in zip(x_values, pmf_values)
	 println("P(X = $x) = $pmf")
end



#####################################
#####################################
#####################################


############################
##### 学習データの生成 #####
############################
# サンプル数
N = 100

# ベルヌーイ分布のインスタンス生成
bernoulli_dist_data = Bernoulli(mu_truth)

# ベルヌーイ分布に従う乱数でデータを生成
data_samples = rand(bernoulli_dist_data, N)

# 0,1の生成回数の確認
counts = countmap(data_samples)
println("$counts")

############################
############################
############################



##########################
##### 事前分布の設定 #####
##########################
# 超パラメータの設定
a = 0.5
b = 0.5

# muの設定
mu = 0:0.001:1

# 事前ベータ分布のインスタンスの生成
beta_dist = Beta(a,b)

# 確率密度関数(PDF)の計算
beta_density = pdf(beta_dist, mu)



##########################
##### 事後分布の生成 #####
##########################
# データの成功回数(1)と失敗回数(0)
N_successes = counts[1]
N_failures  = counts[0]

# 事後分布のパラメータ
a_post = a + N_successes
b_post = b + N_failures

# 事後ベータ分布のインスタンスの生成
posterior_dist = Beta(a_post, b_post)

# 事後分布の確率密度関数(PDF)の計算
posterior_density = pdf(posterior_dist, mu)
#plot(xlim=(0,1), ylim=(0,8))
#plot!(mu,posterior_density, label="")
#vline!([mu_truth], label="", color=:red)
#annotate!(0.9, 7.5, text("N = $N", :right, 12))
#savefig("betaposterior_a05_b05_N50.png")



##########################
##### 予測分布の生成 #####
##########################
# 予測分布のパラメータ
mu_star = mean(posterior_dist)
println("$mu_star")

# 予測分布のインスタンス
bernoulli_dist_posterior = Bernoulli(mu_star)

# 確率質量関数(PMF)の計算
x_star_values = [0, 1]
pmf_star_values = [pdf(bernoulli_dist_posterior, x) for x in x_star_values]

# PMFの確認
for (x, pmf) in zip(x_star_values, pmf_star_values)
	 println("P(X = $x) = $pmf")
end
#bar(x_values, pmf_values, linestyle=:dash, label="true", linecolor=:red, alpha = 0.1)
bar(x_values, pmf_star_values, label="predict", linecolor =:blue, alpha = 0.1)
hline!([mu_truth], label="", color=:red, linestyle=:dash)
savefig("compare_true_predict_N100.png")


##########################
##########################
##########################
