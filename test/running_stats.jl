using Base.Test
using Hobo
using StatsBase

moms(x) = [mean(x), std(x), var(x), skewness(x), kurtosis(x)]

srand(42)
N = 70
data = rand(N)
rs = RunningStats()

push!(rs, data[1])

for i=2:N
    push!(rs, data[i])
    @test_approx_eq moms(data[1:i]) moms(rs)
end


# test + operator
x = rand(500000)
y = x[1:250000]
z = x[250001:end]

rsx = RunningStats(x)
rsy = RunningStats(y)
rsz = RunningStats(z)

@test_approx_eq moms(rsx) moms(rsy + rsz)
