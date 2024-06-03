using Distributions
using Plots
using Random

Random.seed!(123)

# 試行回数を設定
M = 12

# 平均発生回数 λ を設定
parameter = 1.0

# ポアソン分布のインスタンスを生成
poisson_dist = Poisson(parameter)

# サンプル数
x = 0:12

# ポアソン分布からサンプルを生成
y = pdf(poisson_dist, x)

# ヒストグラムを描画
bar(x, y, xlabel="Number of Events", ylabel="Frequency", title="Poisson Distribution", xlims=(0,12))

# 図を保存
savefig("poisson_10.png")
