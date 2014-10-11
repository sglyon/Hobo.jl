module Hobo

using Docile
@docstrings

# TODO: not sure if I should grab first 2 moments from Base or StatsBase
import Base: +, mean, std, var, push!
import StatsBase: mean, std, var, skewness, kurtosis

export csminwel, RunningStats

include("csminwel.jl")
include("running_stats.jl")

end # module
