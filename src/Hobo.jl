module Hobo

# using Docile
# @docstrings

# TODO: not sure if I should grab first 2 moments from Base or StatsBase
import Base: +, mean, std, var, push!, copy
import StatsBase: mean, std, var, skewness, kurtosis

export
    csminwel, RunningMoments, skewness, kurtosis, gensys

include("bayes.jl")
include("csminwel.jl")
include("gensys.jl")
include("running_moments.jl")

end # module
