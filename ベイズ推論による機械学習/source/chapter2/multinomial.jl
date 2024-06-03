using Plots
using Distributions
using PyPlot
using Printf

M = 6
p = [0.15, 0.7, 0.15]
d = Multinomial(M,p)


y = zeros(M,M)
k = zeros(M)
m = zeros(M)

for i in 1:M
	for j in 1:M
		y[i,j] = pdf(d, [i-1, j-1, M - ((i-1) + (j-1))])
	end
end

fig, ax = subplots()
cs = ax.imshow(y, origin = "upper")
fig.colorbar(cs)
plt.savefig("multinomial_02.png")
