module Bayes

using Distributions

export hpd_cs

# highest posterior density credible set
function hpd_cs(d::Distribution, α::Real=0.95, N::Int=1500)
    m = mode(d)

    lb = 1e-9
    ub = 1.0 - lb
    a = quantile(d, lb)
    b = quantile(d, ub)

    x = linspace(a, b, N)

    p_x = pdf(d, x)

    total_p = sum(p_x)

    # start somewhere really high
    p_star = 100.0

    step = 0.01
    converged = false

    high_p_x = similar(p_x, Bool)

    while !converged
        p_star -= step

        # indices for x values above p_star
        high_p_x = p_x .>= p_star

        # converged if weight above p_star over total weight >= α

        converged = sum(p_x[high_p_x]) / total_p >= α
    end

    con_ints = connected_intervals(high_p_x)
    out = map(i->(getindex(x, i[1]), getindex(x, i[end])), con_ints)
    return out
end


function is_disjoint(x::Union(Vector{Bool}, BitArray))
    a_i = findfirst(x)
    ans = false  # assume connectivity
    for j=a_i:size(x, 1)
        if x[j] == true
            continue
        end

        # now x[j] == false
        if sum(x[j:end]) == 0
            # if sum of remainder of array is zero, all elements are false
            # and we are connected
            ans = true
            break
        end
    end
    return ans
end


function first_connected_interval(x::Union(Vector{Bool}, BitArray), starting::Int=1)
    first_t = findnext(x, starting)

    find_next_last_t() = findnext(i->!(i), x, first_t+1) - 1

    if !is_disjoint(x)
        # connected array. If last element is true, all on first_t:end are
        # true. Otherwise, compute manually
        next_last_t = x[end] == true ? size(x, 1) : find_next_last_t()
    else
        next_last_t = find_next_last_t()
    end
    return first_t:next_last_t
end

# Returns Vector of ranges
function connected_intervals(x::Union(Vector{Bool}, BitArray))
    out = Any[]
    cur_last = 0

    if x[end] == true
        last_true = size(x, 1)
    else
        last_true = size(x, 1) - (findfirst(reverse(x)) - 1)
    end

    while cur_last != last_true
        cur_interval = first_connected_interval(x, cur_last + 1)
        cur_last = cur_interval[end]
        push!(out, cur_interval)
    end
    return out
end

end
