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
savefig("model_dist_accuaracy.png")

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
savefig("data_dist_accuracy.png")

############################
############################
############################


####################
##### 事前分布 #####
####################
# 超パラメータ
a = 1
b = 1

# 事前ガンマ分布のインスタンス
prior_dist = Gamma(a, 1/b)

# 確率密度関数
x_values = 0:0.0001:mu_true * 4
pdf_prior = pdf.(prior_dist, x_values)
plot(x_values, pdf_prior, xlim=(0,0.04), ylim=(0.96,1))
savefig("prior_dist_accuracy.png")

####################
####################
####################



####################
##### 事後分布 #####
####################
# パラメータ
a_hat = a + N * 0.5
diff = [(data_samples[i] - mu_true)^2 for i in 1:N]
b_hat = 0.5 * sum(diff) + b
println(a_hat)
println(b_hat)

# 事後ガンマ分布のインスタンス
posterior_dist = Gamma(a_hat, 1/b_hat)

# 確率密度関数
pdf_posterior = pdf.(posterior_dist, x_values)
plot(x_values, pdf_posterior, label="posterior", xlim=(0,0.04))
vline!([lambda_true], label="True")
savefig("posterior_dist_accuracy.png")

####################
####################
####################



####################
##### 予測分布 #####
####################
# パラメータ
mu_s = mu_true
lambda_s = a_hat / b_hat
sigma_s = 1.0 / sqrt(lambda_s)
nu_s = 2 * a_hat

# 予測分布のインスタンス
prior_dist = TDist(nu_s)

# スケールとシフトを加えた確率密度
x_range = mu_true - 4 * sqrt(1/lambda_true):0.01:mu_true + 4 * sqrt(1/lambda_true)
scale = [(x - mu_s) / sigma_s for x in x_range]
pdf_predict = [ pdf(prior_dist, s) / sigma_s for s in scale]
println(length(x_range))
println(length(scale))
println(length(pdf_predict))
println(length(pdf_true))

plot(x_range, pdf_predict, label="predict")
plot!(x_range, pdf_true, label="model", color=:red, linestyle=:dash)
savefig("predict_dist_accuracy.png")
