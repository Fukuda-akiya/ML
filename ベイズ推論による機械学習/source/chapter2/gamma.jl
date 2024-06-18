using Distributions
using Plots

# パラメータベクトル
a = [1.0, 2.0, 2.0]
b = [1.0, 2.0, 0.5]

# ガンマ分布のインスタンス作成
dist = [Gamma(a[i], b[i]) for i in 1:length(a)]

# xの範囲
x = 0:0.01:10

# グラフのタイトル etc
plot(title="Gamma Distributions", xlabel="λ", ylabel="Density") 
plot!(xlim=(0,10), ylim=(0.0,1.2))

# PDFの計算とplot
for i in 1:length(dist)
	y = pdf.(dist[i],x)
	plot!(x, y, label="a=$(a[i]), b=$(b[i])")
end 

savefig("gamma_dist.png")
