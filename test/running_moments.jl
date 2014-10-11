using Base.Test
using Hobo
using StatsBase

moms(x) = [mean(x), std(x), var(x), skewness(x), kurtosis(x)]

srand(42)
N = 70
data = rand(N)
run_m = RunningMoments()

push!(run_m, data[1])

for i=2:N
    push!(run_m, data[i])
    @test_approx_eq moms(data[1:i]) moms(run_m)
end


# test + operator
x = rand(500000)
y = x[1:250000]
z = x[250001:end]

run_mx = RunningMoments(x)
run_my = RunningMoments(y)
run_mz = RunningMoments(z)

@test_approx_eq moms(run_mx) moms(run_my + run_mz)
