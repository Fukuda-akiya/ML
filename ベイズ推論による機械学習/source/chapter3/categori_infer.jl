using Distributions
using Plots
using Random
using StatsBase

Random.seed!(123)


########################
##### モデルの生成 #####
########################
# 次元数
K = 3

# 真のパラメータ
pi_truth = [0.3, 0.5, 0.2]

# カテゴリ分布のインスタンス生成
categorical_dist_true = Categorical(pi_truth)

# 確率質量関数（PMF）の生成
pmf_true = [pdf(categorical_dist_true, i) for i in 1:K]
for i in 1:K
	println("カテゴリ $i の確率: ", pmf_true[i])
end

########################
########################
########################


############################
##### 学習データの生成 #####
############################
# サンプル数
N = 5000

# カテゴリ分布に従うデータの生成
data_samples = rand(categorical_dist_true, N)

# 生成したデータの確認
counts = countmap(data_samples)
println("$counts")

########################
########################
########################



##########################
##### 事前分布の設定 #####
##########################
# 超パラメータの設定
a_k = [1.0, 1.0, 1.0]

# 確率グリッドを生成
x = 0:0.001:1
y = 0:0.001:1
n_points = Int(1 / 0.001)

# 事前ディリクレ分布のインスタンスを生成
prior_dist = Dirichlet(a_k)

# 確率密度関数（PDF）の計算
z = zeros(n_points+1, n_points+1)
for i in 1:n_points+1
	for j in 1:n_points+1
		if x[i] + y[j] <= 1
			z[i, j] = pdf(prior_dist, [x[i], y[j], 1 - x[i] - y[j]])
		else
			z[i, j] = NaN  # ディリクレ分布の範囲外の点
		end
	end
end

#plot(x, y, z, title="Dirichlet Distribution PDF", xlabel="x", ylabel="y",  st = :surface, zlim=(0,25),clim=(0,100))
#savefig("prior_dist.png")

##########################
##########################
##########################



##########################
##### 事後分布の生成 #####
##########################
# 事後分布のパラメータ
a_k_hat = [(counts[i] + a_k[i]) for i in 1:K]

# 事後ディリクレ分布のインスタンスを生成
posterior_dist = Dirichlet(a_k_hat)

# 確率密度を計算
z_posterior = zeros(n_points+1, n_points+1)
for i in 1:n_points+1
	for j in 1:n_points+1
		if x[i] + y[j] <= 1
			z_posterior[i, j] = pdf(posterior_dist, [x[i], y[j], 1 - x[i] - y[j]])
		else
			z_posterior[i, j] = NaN  # ディリクレ分布の範囲外の点
		end
	end
end

#plot(x, y, z_posterior, title="Dirichlet Distribution PDF", xlabel="x", ylabel="y",  st = :surface, color=:rainbow)
#savefig("posterior_dist.png")

##########################
##########################
##########################


##########################
##### 予測分布の生成 #####
##########################
# 予測分布のパラメータ
pi_star = mean(posterior_dist)

# 予測分布のインスタンスを生成
predict_dist =  Categorical(pi_star)

# 確率質量関数（PMF）の生成
pmf_predict = [pdf(predict_dist, i) for i in 1:K]
for i in 1:K
	println("カテゴリ $i の確率: ", pmf_predict[i])
end

k_values = 1:1:3
bar(k_values, pmf_predict, label="predict", alpha=0.5)
hline!([pmf_true[1]], label="1", color=:red, linestyle = :dash)
hline!([pmf_true[2]], label="2", color=:orange, linestyle = :dash)
hline!([pmf_true[3]], label="3", color=:yellow, linestyle = :dash)
plot!(legend=:topright)
savefig("compare_ture_predict_N5000.png")
