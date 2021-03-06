"""
Implements a type that computes the running mean, variance, standard
deviation, skewness, and kurtosis for a data set. Follows the algorithm
outlined [here](http://www.johndcook.com/skewness_kurtosis.html)
"""

type RunningMoments{T <: Number}
    M1::T
    M2::T
    M3::T
    M4::T
    n::Int
end

function RunningMoments()
    return RunningMoments(0.0, 0.0, 0.0, 0.0, 0)
end

function RunningMoments(x::Number)
    rs = RunningMoments()
    push!(rs, x)
    return rs
end

function RunningMoments(x::Vector)
    rs = RunningMoments()
    push!(rs, x)
    return rs
end

function push!(rs::RunningMoments, x::Number)
    n1 = rs.n
    rs.n += 1

    delta = x - rs.M1
    delta_n = delta / rs.n
    delta_n2 = delta_n * delta_n
    term1 = delta * delta_n * n1
    rs.M1 += delta_n
    rs.M4 += term1*delta_n2*(rs.n*rs.n - 3rs.n + 3) + 6*delta_n2*rs.M2 -
             4*delta_n*rs.M3
    rs.M3 += term1*delta_n * (rs.n - 2) - 3*delta_n*rs.M2
    rs.M2 += term1
    nothing
end

function push!(rs::RunningMoments, x::Vector)
    n = size(x, 1)
    for i=1:n
        @inbounds push!(rs, x[i])
    end
    nothing
end

mean(rs::RunningMoments) = rs.M1
var(rs::RunningMoments) = rs.M2/(rs.n - 1.0)
std(rs::RunningMoments) = sqrt(var(rs))
skewness(rs::RunningMoments) = sqrt(rs.n) * rs.M3 / rs.M2^(1.5)
kurtosis(rs::RunningMoments) = rs.n*rs.M4 / (rs.M2*rs.M2) - 3.0

function +(a::RunningMoments, b::RunningMoments)
    rs = RunningMoments()
    rs.n = a.n + b.n

    delta = b.M1 - a.M1
    delta2 = delta*delta
    delta3 = delta*delta2
    delta4 = delta2*delta2

    rs.M1 = (a.n*a.M1 + b.n*b.M1) / rs.n

    rs.M2 = a.M2 + b.M2 + delta2 * a.n * b.n / rs.n

    rs.M3 = a.M3 + b.M3 + delta3 * a.n * b.n * (a.n - b.n)/(rs.n*rs.n)
    rs.M3 += 3.0*delta * (a.n*b.M2 - b.n*a.M2) / rs.n

    rs.M4 = a.M4 + b.M4 + delta4*a.n*b.n * (a.n*a.n - a.n*b.n + b.n*b.n) /
                  (rs.n*rs.n*rs.n)
    rs.M4 += 6.0*delta2 * (a.n*a.n*b.M2 + b.n*b.n*a.M2)/(rs.n*rs.n) +
                  4.0*delta*(a.n*b.M3 - b.n*a.M3) / rs.n

    return rs
end


copy(rm::RunningMoments) = RunningMoments() + rm
