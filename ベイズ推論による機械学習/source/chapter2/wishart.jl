using Distributions
using Plots
using Printf

# スケール行列Wを定義
W = [0.5 0.0; 0.0 0.5]

# 自由度nを設定
n = 20

# ウィシャート分布のインスタンスを作成
w_dist = Wishart(n, W)

# サンプリング
sample_matrix1 = rand(w_dist)
sample_matrix2 = rand(w_dist)
sample_matrix3 = rand(w_dist)
sample_matrix4 = rand(w_dist)
sample_matrix5 = rand(w_dist)

# 2次元ガウス分布のパラメータ
mu = [0, 0] # 平均ベクトル

# ガウス分布のインスタンス作成
g_dist1 = MvNormal(mu, sample_matrix1)
g_dist2 = MvNormal(mu, sample_matrix2)
g_dist3 = MvNormal(mu, sample_matrix3)
g_dist4 = MvNormal(mu, sample_matrix4)
g_dist5 = MvNormal(mu, sample_matrix5)

# 変数
x = -2:0.01:2
y = -2:0.01:2

# 確率密度関数
z1 = [pdf(g_dist1, [i, j]) for i in x, j in y]
#z2 = [pdf(g_dist2, [i, j]) for i in x, j in y]
#z3 = [pdf(g_dist3, [i, j]) for i in x, j in y]
#z4 = [pdf(g_dist4, [i, j]) for i in x, j in y]
#z5 = [pdf(g_dist5, [i, j]) for i in x, j in y]

# 描画
contour!(x, y, z1, fill=false)
#contour!(x, y, z2, levels=[0.02], linewidth=2, fill=false)
#contour!(x, y, z3, levels=[0.02], linewidth=2, fill=false)
#contour!(x, y, z4, levels=[0.02], linewidth=2, fill=false)
#contour!(x, y, z5, levels=[0.02], linewidth=2, fill=false)
savefig("wishart_make_gauss_sample03.png")
