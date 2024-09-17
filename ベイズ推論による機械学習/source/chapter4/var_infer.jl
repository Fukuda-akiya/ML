using Plots
using Distributions
using Random
using StatsBase
using SpecialFunctions

Random.seed!(111)

##################
##### モデル #####
##################
# モデルはポアソン分布に従う
# モデルのクラスタリングはカテゴリ分布に従う

# 正解のlambdaを指定
lambda_truth_k = [10.0, 25.0, 40.0]

# 正解のpiを指定
pi_truth_k = [0.35, 0.25, 0.4]

# クラスタ数を取得
K = length(lambda_truth_k)

# x のグリッド点を準備
x_line = 0:1:2 * maximum(lambda_truth_k)

# モデルの分布
model_dist = [Poisson(lambda_truth_k[i]) for i in 1:K]

# モデルの重み付きPDFを計算
model_prob = zeros(length(x_line))
for k in 1:K
	# 各クラスタのPDFを計算
	tmp_pdf = pdf.(model_dist[k],x_line)

	# 各クラスタをpiで重み付け
	global model_prob
	model_prob += pi_truth_k[k] * tmp_pdf
end

# グラフ描画
bar(x_line, model_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model")
savefig("model_data.png")


######################
##### データ生成 #####
######################

# サンプルデータ数
N = 250

# 正解のs_nkを作成
s_truth_nk = rand(Categorical(pi_truth_k), N)

# 1 of K 表現に変換
one_hot_s_truth_nk = zeros(Int, N, K)
for i in 1:N
	    one_hot_s_truth_nk[i, s_truth_nk[i]] = 1
end

# データの生成
x_n = [rand(Poisson(lambda_truth_k[s_truth_nk[i]])) for i in 1:N]

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

# lambdaの事前分布の確率密度を計算
lambda_line = range(0, stop=2.0*maximum(lambda_truth_k), length=1000)
pdf_lambda_prior = pdf(Gamma(a,1/b), lambda_line)

# lambdaの事前分布の描画
plot(lambda_line, pdf_lambda_prior, xlabel="lambda", ylabel="PDF", title="Prior Dist PDF", label="prior dist")
vline!(lambda_truth_k, label="True val", ls=:dash)
savefig("prior_dist.png")

# piの事前分布のパラメータ
alpha_k = fill(2.0, K)

############################################
##### lambdaとpiの近似事後分布を初期化 #####
############################################

# 潜在変数の近似事後分布の期待値をランダムに生成し、初期値とする
E_s_nk = rand(N, K)
E_s_nk ./= sum(E_s_nk, dims=2) # 正規化

# lambdaの近似事後分布を初期化
a_hat_k = sum(E_s_nk .* x_n[:], dims=1) .+ a
b_hat_k = sum(E_s_nk, dims=1) .+ b
#E_lambda_k = rand([Gamma(a_hat_k[k],1/b_hat_k[k]) for k in 1:K],K)

# lambdaの初期近似分布を描画
post_lambda_kl = zeros(K, length(lambda_line))
#plot(title="lambda initial distributions")
#for k in 1:K
#	post_lambda_kl[k,:] = pdf.(Gamma(a_hat_k[k],1/b_hat_k[k]), lambda_line)
#	plot!(lambda_line, post_lambda_kl[k, :], label="Cluster$k")
#end	
#vline!(lambda_truth_k, label="truth lambda")
#savefig("lambda_appro_init.png")

# piの近似事後分布を初期化
alpha_hat_k = vec(sum(E_s_nk, dims=1)) .+ alpha_k

# 生成した近似分布のパラメータを用いて、混合分布を計算する
E_lambda_k = a_hat_k ./ b_hat_k
E_pi_k = alpha_hat_k ./ sum(alpha_hat_k, dims=1)

# 初期値による混合分布を計算
init_prob = zeros(length(x_line))
init_dist = [Poisson(E_lambda_k[k]) for k in 1:K] 
for k in 1:K
	 # クラスタkの分布の確率を計算
	 tmp_pdf = pdf.(init_dist[k],x_line)

	 # K個の分布の加重平均を計算
	 global init_prob
	 init_prob += E_pi_k[k] * tmp_pdf
end

# グラフ描画
#bar(x_line, init_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model", label="Init prob")
#bar!(x_line, model_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model", label="modeldata",ls=:dash,alpha=0.5)
#savefig("init_poisson_model.png")

####################
##### 変分推論 #####
####################

# サンプリング回数
iterator = 50

eta_nk = zeros(N,K)
s_nk = zeros(N,K)
one_hot_s_nk = zeros(Int, N, K) 

save_lambda_k = zeros(iterator, K)
save_pi_k = zeros(iterator, K)
save_a_hat_k = zeros(iterator, K)
save_b_hat_k = zeros(iterator, K)
save_alpha_hat_k = zeros(iterator, K)

for i in 1:iterator
	# 潜在変数の近似事後分布のパラメータに関する期待値を更新
	global E_lambda_k
	E_lambda_k = vec(a_hat_k ./ b_hat_k)
	global E_ln_lambda_k
	E_ln_lambda_k = vec(digamma.(a_hat_k) .- log.(b_hat_k))
	global E_ln_pi_k
	E_ln_pi_k = digamma.(alpha_hat_k) .- digamma.(sum(alpha_hat_k, dims=1))
	
	#############################################
	##### s_nの近似事後分布のパラメータ更新 #####
	#############################################
	for n in 1:N
		# 潜在変数の近似事後分布のパラメータを更新
		tmp_eta_k = exp.(x_n[n] * E_ln_lambda_k .- E_lambda_k .+ E_ln_pi_k)
		eta_nk[n,:] = tmp_eta_k ./ sum(tmp_eta_k)
	 	
		E_s_nk[n,:] = eta_nk[n,:]	
	end

	##################################################
	##### lambdaの近似事後分布のパラメータの更新 #####
	##################################################
	global a_hat_k, b_hat_k
	a_hat_k = sum(E_s_nk .* x_n, dims=1) .+ a
	b_hat_k = sum(E_s_nk, dims=1) .+ b
	save_a_hat_k[i,:] = a_hat_k
	save_b_hat_k[i,:] = b_hat_k
	global lambda_samp_k = zeros(K)
	lambda_samp_k = [rand(Gamma(a_hat_k[k], 1/b_hat_k[k])) for k in 1:K]
	save_lambda_k[i,:] = lambda_samp_k

	##############################################
	##### piの近似事後分布のパラメータの更新 #####
	##############################################
	global alpha_hat_k
	alpha_hat_k = vec(sum(E_s_nk, dims=1)) .+ alpha_k
	save_alpha_hat_k[i, :] = alpha_hat_k

	# piのサンプル
	global pi_samp_k = zeros(K)
	pi_samp_k = rand(Dirichlet(alpha_hat_k))
	save_pi_k[i,:] = pi_samp_k
end

# 最終的に得られたlambdaの確認
post_lambda_kl = zeros(K, length(lambda_line))
plot(title="lambda final distributions")
for k in 1:K
	post_lambda_kl[k,:] = pdf.(Gamma(a_hat_k[k],1/b_hat_k[k]), lambda_line)
	plot!(lambda_line, post_lambda_kl[k, :], label="Cluster$k")
end	
vline!(lambda_truth_k, label="truth lambda")
savefig("lambda_appro_final.png")


# 更新したパラメータによる混合分布を計算
E_pi_k = vec(alpha_hat_k ./ sum(alpha_hat_k, dims=1))
final_prob = zeros(length(x_line))
final_dist = [Poisson(E_lambda_k[k]) for k in 1:K] 
for k in 1:K
	 # クラスタkの分布の確率を計算
	 tmp_pdf = pdf.(final_dist[k],x_line)

	 # K個の分布の加重平均を計算
	 global final_prob
	 final_prob += E_pi_k[k] * tmp_pdf
end

# グラフ描画
bar(x_line, final_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model", label="Final prob")
bar!(x_line, model_prob, xlabel="x", ylabel="prob", title="Poisson Mixture Model", label="modeldata",ls=:dash,alpha=0.5)
savefig("Final_poisson_model.png")


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

plot(title="alpha_hat transition", xlabel="Iterarion", ylabel="alpha_hat values")
[plot!(tried_num, [save_alpha_hat_k[:,k]], label="Cluster0$k") for k in 1:K]
savefig("alpha_hat_transition.png")
