using Distributions
using Plots
using Random
using LinearAlgebra

Random.seed!(123)

# 入力ベクトルの準備
function x_vector(x_n, m)
    	# 行列 x_nm の初期化
        x_nm = zeros(m, length(x_n))
	# m乗を計算して行列に格納
	for i in 1:m
		for j in 1:length(x_n)
			x_nm[i, j] = x_n[j]^(i-1)
		end
	end					        
	return x_nm
end

# 多項式の次元数
m = 4
# 関数ベクトル
f = [1]
# 入力行列
x_nm = x_vector(f, m)
#display(x_nm)

########################
##### モデルの準備 #####
########################

# モデルにはノイズがないものと仮定

# 真の多項式の次元数
m_truth = 4

# パラメータwの準備
w_truth = rand(-1:0.1:1, m_truth)
#println(w_truth)
#w_truth = [-0.5 -0.8  0.3  0.2]

# グラフ用にxの値域を設定
x_values = -3.0:0.01:3.0

# M次元に拡張
x_truth_nm = x_vector(x_values,m)
#println(x_truth_nm[:,1:5])

# 真のモデルの出力
y_line = w_truth' * x_truth_nm
#println(y_line[1:5])

# グラフを出力
#plot(x_values, y_line[1,:], label="True")
#savefig("model_graph.png")


########################
########################
########################


#####################################
##### データの生成（ノイズあり）#####
#####################################

# ノイズはガウス分布にしたがっているとする
# 平均パラメータ
mu = 0.0
# 標準偏差
sigma = 1.5
# 精度パラメータ
lmd = 1.0 / sigma^2

# データを生成する
# 観測データ数（訓練データ数）
N = 10

# 入力値の生成（乱数）
x_n = rand(minimum(x_values):0.01:maximum(x_values), N)

# 入力値をm次元に拡張
x_truth_nm = x_vector(x_n, m_truth)

# 1次元ガウス分布に従うノイズ成分を生成
noise_dist = Normal(mu,sqrt(1/lmd))
epsilon_n = rand(noise_dist, N)
y_n = reshape(w_truth' * x_truth_nm, (N)) + epsilon_n

# モデルと生成したデータの比較
#plot(x_values, y_line[1,:], label="True")
#scatter!(x_n,y_n, label="Sample")
#savefig("compare_model_sample.png")

####################
##### 事前分布 #####
####################
# パラメータ w の学習
# 事前分布の次元数
m = 4

# 事前分布の平均
m_m = zeros(m)

# 事前分布の精度行列を指定
# 共分散行列（対角行列と仮定）
sigma_mm = Matrix(I,m,m) * 10^2
# 精度行列（共分散行列の逆行列）
lmd_mm = inv(sigma_mm) 

# 事前ガウス分布のインスタンス生成
prior_dist = MvNormal(m_m, sigma_mm)

# 入力値をM次元に拡張
x_nm = x_vector(x_n, m)
x_arr = x_vector(x_values, m)

# サンプリング数
samp_num = 5
#plot(x_values, y_line[1,:], label="True", ls=:dash)
# 事前分布からサンプリングしたwを用いたモデルを比較
prior_list = []
for i in 1:samp_num

	# パラメータの生成
	w_prior = rand(prior_dist, 1)
		
	# 出力値
	y_samp = vec(w_prior' * x_arr)
	
	push!(prior_list, y_samp)
end
#plot!(x_values, prior_list)
#plot!(ylim=(-5,10))
#savefig("w_samp_prior.png")



####################
##### 事後分布 #####
####################
# パラメータ
lmd_hat_mm = lmd * x_nm * x_nm' + lmd_mm
inv_lmd_hat_mm = inv(lmd_hat_mm)
term_01 = lmd * reshape(y_n, (1, N)) * x_nm'
term_02 = lmd_mm * m_m
m_hat_m = inv(lmd_hat_mm) * (vec(term_01) + term_02)

# 事後ガウス分布のインスタンス
##### Float32, つまり6桁の数値精度にしないと lmd_hat_mm がエルミート行列だと認識されないのでFloat64からFloat32に変換 #####
# juliaでは逆行列に対してコレフスキー分解が上手く適用できない（数値計算では逆行列は御法度？）
eig_vals, eig_vecs = eigen(inv_lmd_hat_mm)
eig_vals = max.(eig_vals, 1e-6)
inv_lmd_hat_mm_hermitian = eig_vecs * Diagonal(eig_vals) * eig_vecs'
post_dist = MvNormal(Float32.(m_hat_m), Float32.(0.5 * (inv_lmd_hat_mm_hermitian)))

#plot(x_values, y_line[1,:], label="True", ls=:dash)
# 事後分布から w をサンプリング 
post_list = []
for i in 1:samp_num
	# パラメータの生成
	w_post = rand(post_dist, 1)
	
	# 出力値
	y_samp_post = vec(w_post' * x_arr)
	 
	push!(post_list, y_samp_post)
end
#scatter!(x_n,y_n, label="Sample")
#plot!(x_values, post_list)
#savefig("w_samp_post.png")



####################
##### 予測分布 #####
####################
# パラメータ
mu_star = vec(m_hat_m' * x_arr)
term_01 = x_arr' * inv(lmd_hat_mm) * x_arr
lmd_star = vec(diag(term_01) .+ lmd^-1)
#display(sqrt.(lmd_star).^-1)

# 予測分布のインスタンス
predict_dist = [Normal(mu_star[i], sqrt.(lmd_star[i]).^-1) for i in 1:length(mu_star)]

# 出力値 y の推論(乱数によるサンプリング)
y_star = [mean([rand(predict_dist[i]) for _ in 1:N]) for i in 1:length(predict_dist)]

plot(x_values, y_line[1,:], label="True", ls=:dash)
scatter!(x_n,y_n, label="Model")
#plot!(x_values, y_star, label="predict rand")
# 期待値
plot!(x_values, y_star, label="predict mean", color=:red)
# 分散
plot!(x_values, y_star, ribbon=sqrt.(lmd_star), fillalpha=0.1, color="#00A968", label="μ ± σ")
plot!(ylim=(-15,25))
savefig("predict_N10_M4.png")
