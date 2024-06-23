using Distributions
using Plots
using Random

Random.seed!(123)


########################
##### モデルの構築 #####
########################
# 真のパラメータ
mu_true = 25
lambda_true = 0.01

# モデルガウス分布のインスタンス
model_dist = Normal(mu_true, sqrt(1/lambda_true))

# 確率密度関数
x_values = mu_true - 4 * sqrt(1/lambda_true):0.01:mu_true + 4 * sqrt(1/lambda_true)
pdf_true = pdf(model_dist, x_values)

plot(x_values, pdf_true)
savefig("model_dist.png")

########################
########################
########################



############################
##### 学習データの生成 #####
############################
# サンプル数
N = 50

# ガウス分布のインスタンス
data_dist = Normal(mu_true, sqrt(1/lambda_true))

# データの生成
data_samples = rand(data_dist, N)
bar(data_samples)
savefig("data_dist.png")

############################
############################
############################


####################
##### 事前分布 #####
####################
# 超パラメータ
mu_prior = 0
lambda_mu = 0.001

# 事前ガウス分布のインスタンス
prior_dist = Normal(mu_prior, sqrt(1/lambda_mu))

# 確率密度関数
mu_values = mu_true - 50:0.01:mu_true + 50
pdf_prior = pdf.(prior_dist, mu_values)
plot(mu_values, pdf_prior)
savefig("prior_dist.png")

####################
####################
####################



####################
##### 事後分布 #####
####################
# パラメータ
lambda_mu_hat = N * lambda_true + lambda_mu
mu_hat = ( lambda_true * sum(data_samples) + lambda_mu * mu_prior)/lambda_mu_hat

println(lambda_mu_hat)
println(mu_hat)

# 事後ガウス分布のインスタンス
posterior_dist = Normal(mu_hat, sqrt(1/lambda_mu_hat))

# 確率密度関数
pdf_posterior = pdf.(posterior_dist, mu_values)
plot(mu_values, pdf_posterior, label="posterior")
vline!([mu_true], label="True")
savefig("posterior_dist.png")

####################
####################
####################



####################
##### 予測分布 #####
####################
# パラメータ
lambda_star = (lambda_true * lambda_mu_hat) / (lambda_true + lambda_mu_hat)
mu_star = ( lambda_true * sum(data_samples) + lambda_mu * mu_prior)/(lambda_mu + N * lambda_true)

# 予測分布のインスタンス
prior_dist = Normal(mu_star, sqrt(1/lambda_star))

# 確率密度関数
pdf_predict = pdf.(prior_dist, x_values)
plot(x_values, pdf_predict, label="predict")
plot!(x_values, pdf_true, label="model", color=:red, linestyle=:dash)
savefig("predict_dist.png")
