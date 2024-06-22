using Distributions
using Plots
using Random
using StatsBase

Random.seed!(111)

########################
##### モデルの生成 #####
########################
# 真のパラメータを設定
lambda_true = 4.0

# ポワソン分布のインスタンスを生成
poisson_dist_true = Poisson(lambda_true)

# 確率質量関数の計算
x_values = 0:1:16
pmf_true = pdf(poisson_dist_true, x_values)

# モデル構造の確認
#bar(x_values, pmf_true)
#savefig("model_dist.png")

########################
########################
########################


############################
##### 学習データの生成 #####
############################
# サンプル数
N = 50

# ポワソン分布のインスタンスを生成
poisson_dist_data = Poisson(lambda_true)

# ポワソン分布に従うデータを乱数で生成
data_samples = rand(poisson_dist_data, N)
counts = countmap(data_samples)
pmf_data = [get(counts, x, 0)/N for x in x_values]

# 生成したデータの確認
#bar(counts)
#bar(x_values, pmf_data)
#savefig("poisson_data.png")

############################
############################
############################



##########################
##### 事前分布の生成 #####
##########################
# 超パラメータの設定
a = 1
b = 1

# 事前ガンマ分布のインスタンスを生成
prior_dist = Gamma(a,b)

# 確率密度関数の計算
lambda_values = 0:0.001:8
pdf_prior = pdf(prior_dist, lambda_values)
#plot(lambda_values, pdf_prior)
#savefig("prior_dist.png")

##########################
##########################
##########################


##########################
##### 事後分布の生成 #####
##########################
# 事後分布のパラメータ
a_hat = a + sum(data_samples)
b_hat = b + N

println(a_hat)
println(b_hat)

# 事後ガンマ分布のインスタンスを生成
posterior_dist = Gamma(a_hat,b_hat)

# 確率密度関数
pdf_posterior = pdf(posterior_dist, lambda_values)
plot(lambda_values, pdf_posterior)
vline!([lambda_true], color=:red, linestyle=:dash)
savefig("posterior_dist.png")
