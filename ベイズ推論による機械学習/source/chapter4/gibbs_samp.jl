using Distributions
using Plots
using Random
using StatsBase

Random.seed!(111)



######################
##### 観測モデル #####
######################
# 観測モデルはポアソン分布に従う
# 観測モデルの割り当てはカテゴリ分布に従う

##### 初期値としてlambdaとpiを設定する
# K 個の真のパラメータを指定
lambda_truth_k = [10.0, 25.0, 40.0]

# 真の混合比率を設定
pi_truth_k = [0.35, 0.25, 0.4]

# クラスタ数を取得
K = length(lambda_truth_k)

# x のグリッド点を準備
x_line = 0:1:2*maximum(lambda_truth_k)

# 観測モデルの分布のインスタンス
obs_dist = [Poisson(lambda_truth_k[i]) for i in 1:K]

# 観測モデルを計算
model_prob = zeros(length(x_line))

for i in 1:K
	# クラスタ K の分布の確率を計算
	tmp_prob = pdf.(obs_dist[i], x_line)

	# K 個の分布の加重平均を計算（各分布を混合比率で重み付け）
	global model_prob
	model_prob += pi_truth_k[i] * tmp_prob
end

# グラフ描画
bar(x_line, model_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model")
savefig("obs_model.png")



######################
##### データ生成 #####
######################

# モデルにしたがってデータを生成する
# 最初に潜在変数を各クラスタに対して生成する

# 観測データ数
N = 250

# s_nkのしたがう分布のインスタンス
cluster_dist = Categorical(pi_truth_k)

# 真のクラスタを生成
s_truth_nk = rand(cluster_dist,N)

# 1 of K 表現に変換
one_hot_s_truth_nk = zeros(Int, N, K)
for i in 1:N
	    one_hot_s_truth_nk[i, s_truth_nk[i]] = 1
end

# データ生成のインスタンス
data_dist = [Poisson(lambda_truth_k[i]) for i in s_truth_nk]

# データの生成
x_n = [rand(data_dist[i]) for i in 1:N]

# 生成したデータの出現回数をcountmapで取得し、正規化
data = countmap(x_n)
norm_data = Dict(k => v / N for (k, v) in data)
y_data = collect(values(norm_data))
x_data = collect(keys(norm_data))

# 各クラスタごとのデータを抽出
cluster_data = Dict()
for k in 1:K
	cluster_data[k] = x_n[s_truth_nk .== k]
end

# 観測データの描画
bar(x_data, y_data, label="obsdata")
bar!(x_line, model_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model", label="modeldata",ls=:dash,alpha=0.5)
savefig("gen_data.png")

# データのクラスタの描画
plot(title="Cluster-wise Data Distribution", xlabel="Data Value", ylabel="Frequency", legend=:topright)
for k in 1:K
	histogram!(cluster_data[k], bins=25, alpha=0.6, label="Cluster $k")
end
savefig("gen_data_cluster.png")



####################
##### 事前分布 #####
####################
# 観測データがポアソン分布でそのパラメータlambdaに対しては共役事前分布であるガンマ分布を用いる

# lambdaの事前分布のパラメータ（ハイパーパラメータ）
a = 1.0
b = 1.0

# lambdaの事前分布のインスタンス
lambda_prior_dist = Gamma(a, 1/b)

# lambdaの事前分布の確率密度を計算
lambda_line = range(0, stop=2.0*maximum(lambda_truth_k), length=1000)
pdf_lambda_prior = pdf(lambda_prior_dist, lambda_line)

# lambdaの事前分布の描画
plot(lambda_line, pdf_lambda_prior, xlabel="lambda", ylabel="PDF", title="Prior Dist PDF", label="prior dist")
vline!(lambda_truth_k, label="True val", ls=:dash)
savefig("prior_dist.png")


####################################
##### lambdaとpiの初期値の設定 #####
####################################
# piのパラメータ
alpha_prior_k = repeat([2.0], inner=K)

# piの事前分布のインスタンス
pi_prior_dist = Dirichlet(alpha_prior_k)
# piの生成
pi_prior_k = rand(pi_prior_dist)
#pi_prior_k = [0.27973663, 0.52242674, 0.19783663]

# lambdaの生成
lambda_prior_k = rand(lambda_prior_dist, K)
#lambda_prior_k = [0.35550085, 0.17254374, 2.04555575]

# 事前パラメータを用いたポアソン分布のインスタンス
prior_data_dist = [Poisson(lambda_prior_k[k]) for k in 1:K]

# 初期値を用いてデータを生成する
init_prob = zeros(length(x_line))
for k in 1:K
	# クラスタ k の分布の確率を計算
	tmp_prob = pdf.(prior_data_dist[k], x_line)
	
	# K 個の分布の重み付け平均
	global init_prob 
	init_prob += pi_prior_k[k] * tmp_prob
end

# 描画
bar(x_line, init_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model", label="prior data")
bar!(x_line, model_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model", label="modeldata",ls=:dash,alpha=0.5)
savefig("prior_data_infer.png")



##############################
##### ギブスサンプリング #####
##############################
# サンプリング回数
iterator = 100

# クラスタに関するパラメータη
eta_nk = zeros(N, K)
s_nk = zeros(N, K)

# 1 of K 表現に直す用の行列		
one_hot_s_nk = zeros(Int, N, K)                                                                                               

# 保存用
save_lambda_k = zeros(iterator, K)
save_pi_k = zeros(iterator, K)
save_a_hat_k = zeros(iterator, K)
save_b_hat_k = zeros(iterator, K)

# ギブスサンプリング
for i in 1:iterator
	# 潜在変数Sのパラメータの更新
	for n in 1:N
		if i == 1
			tmp_eta_k = exp.(x_n[n] * log.(lambda_prior_k) .- lambda_prior_k .+ log.(pi_prior_k))
		else
			tmp_eta_k = exp.(x_n[n] * log.(lambda_samp_k) .- lambda_samp_k .+ log.(pi_samp_k))
		end

		eta_nk[n,:] = tmp_eta_k / sum(tmp_eta_k)

		# 潜在変数Sのサンプル
		samp_dist = Categorical(eta_nk[n,:])
		s_nk[n] = rand(samp_dist)
		# s_nを1 of 表現で表示
		one_hot_s_nk[n, :] .= 0
		one_hot_s_nk[n, Int(s_nk[n])] = 1 
	end
	
	###########################
	###### lambdaのsample #####
	###########################
	# a_hat_kの更新には1 of K 表現を用いる
	global a_hat_k = zeros(K)
	a_hat_k = sum(one_hot_s_nk .* x_n[:], dims=1) .+ a
	save_a_hat_k[i,:] = a_hat_k

	# b_hat_kの更新にはクラスタ番号を用いる
	global b_hat_k = zeros(K)
	sum_one_hot_s_nk = sum(one_hot_s_nk, dims=1)
	b_hat_k = sum_one_hot_s_nk .+ b
	save_b_hat_k[i,:] = b_hat_k

	# lambdaのサンプル
	global lambda_samp_k = zeros(K)
	lambda_samp_k = [rand(Gamma(a_hat_k[k], 1/b_hat_k[k])) for k in 1:K]
	save_lambda_k[i,:] = lambda_samp_k
	
	######################
	##### piのsample ##### 
	######################
	# パラメータ更新
	global alpha_hat_k = zeros(K)
	alpha_hat_k = sum(sum_one_hot_s_nk, dims=1)[:] .+ alpha_prior_k
	
	# piのサンプル
	global pi_samp_k = zeros(K)
	pi_samp_k = rand(Dirichlet(alpha_hat_k))
	save_pi_k[i,:] = pi_samp_k
	
	##### sampleしたlambdaとpiを用いて、潜在変数Sを更新する #####
end

tried_num = 0:iterator-1

# lambdaの推移を可視化
plot(title="lambda transition", xlabel="Iterarion", ylabel="lambda value", legend=false, ylims=(0,45))
plot!(tried_num, [save_lambda_k[:,1]], label="Cluster01")
plot!(tried_num, [save_lambda_k[:,2]], label="Cluster02")
plot!(tried_num, [save_lambda_k[:,3]], label="Cluster03")
hline!(lambda_truth_k, label="True value", ls=:dash)
savefig("lambda_transion.png")



# piの推移を可視化
plot(title="pi transition", xlabel="Iterarion", ylabel="pi value", ylims=(0.0,1.0))
plot!(tried_num, [save_pi_k[:,1]], label="Cluster01")
plot!(tried_num, [save_pi_k[:,2]], label="Cluster02")
plot!(tried_num, [save_pi_k[:,3]], label="Cluster03")
hline!(pi_truth_k, label="True Values", ls=:dash)
savefig("pi_transion.png")

# ハイパーパラメータの推移を可視化
plot(title="a_hat transition", xlabel="Iterarion", ylabel="a_hat values")
[plot!(tried_num, [save_a_hat_k[:,k]], label="Cluster0$k") for k in 1:K]
savefig("a_hat_transition.png")

plot(title="b_hat transition", xlabel="Iterarion", ylabel="b_hat values")
[plot!(tried_num, [save_b_hat_k[:,k]], label="Cluster0$k") for k in 1:K]
savefig("b_hat_transition.png")


# ガンマ事後分布の可視化
pdf_post_dist = [pdf.(Gamma(a_hat_k[k], 1/b_hat_k[k]), lambda_line) for k in 1:K]
plot(title="Gamma PostDist", xlabel="lambda value", ylabel="Density")
[plot!(lambda_line, pdf_post_dist[k], label="Cluster0$k") for k in 1:K]
vline!(lambda_truth_k, ls=:dash, label="True Values")
savefig("gamma_postdist_pdf.png")

# 混合分布の可視化
# 初期値を用いてデータを生成する
final_prob = zeros(length(x_line))
post_data_dist = [Poisson(lambda_samp_k[k]) for k in 1:K] 
for k in 1:K
	# クラスタ k の分布の確率を計算
	tmp_prob = pdf.(post_data_dist[k], x_line)
	
	# K 個の分布の重み付け平均
	global final_prob 
	final_prob += pi_samp_k[k] * tmp_prob
end
# 描画
bar(x_line, final_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model", label="post data")
bar!(x_line, model_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model", label="modeldata",ls=:dash,alpha=0.5)
savefig("post_data_infer.png")
