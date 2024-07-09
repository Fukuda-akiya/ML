using Distributions 
using Plots

# パラメータ
mu_s = [0, 0, 0]
lambda_s = [1.0, 1.0, 4.0]
nu_s = [1.0, 4.0, 1.0]
sigma_s = 1.0 ./ sqrt.(lambda_s)

# スチューデントのt分布のインスタンス
dist = [TDist(nu_s[i]) for i in 1:length(nu_s)]

# スケールとシフトを加えた確率密度
x_range = -4:0.01:4
t_pdf = zeros(length(x_range), length(nu_s))
for i in 1:length(nu_s)
	for j in 1:length(x_range)
		t_pdf[j,i] = pdf.(dist[i], (x_range[j] - mu_s[i])/sigma_s[i]) / sigma_s[i]
	end
end

plot(x_range, t_pdf, ylim=(0,1))
savefig("tdist.png")
