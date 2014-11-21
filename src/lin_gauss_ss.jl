using Distributions
using StateSpace

issquare(x::Matrix) = size(x, 1) == size(x, 2)
issquare(t::TimeVaryingParam) = all(map(issquare), t.mats)

using StateSpace: convert

## -------------------------- ##
#- type: LinearGaussianSSabcd -#
## -------------------------- ##

type LinearGaussianSSabcd
    A::TimeVaryingParam{Matrix}
    B::TimeVaryingParam{Matrix}
    C::TimeVaryingParam{Matrix}
    D::TimeVaryingParam{Matrix}

    # Handle matrix or TimeVaryingParam inputs
    function LinearGaussianSSabcd(A::Union(TimeVaryingParam, Matrix),
                                  B::Union(TimeVaryingParam, Matrix),
                                  C::Union(TimeVaryingParam, Matrix),
                                  D::Union(TimeVaryingParam, Matrix))
        @assert issquare(A)
        @assert issquare(B)
        @assert issquare(D)
        @assert size(A) == size(B)
        @assert size(C, 1) == size(D, 1)
        A, B, C, D = map(x->convert(TimeVaryingParam, x), Any[A, B, C, D])
        return new(A, B, C, D)
    end
end

# Univariate state and data
function LinearGaussianSSabcd(A::Real, B::Real, C::Real, D::Real)
    a = Array(Float64, 1, 1); a[1] = A
    b = Array(Float64, 1, 1); b[1] = B
    c = Array(Float64, 1, 1); c[1] = C
    d = Array(Float64, 1, 1); d[1] = D
    return LinearGaussianSSabcd(a, b, c, d)
end

# Univariate state, n-d data
function LinearGaussianSSabcd(A::Real, B::Real, C::Matrix, D::Matrix)
    a = Array(Float64, 1, 1); a[1] = A
    b = Array(Float64, 1, 1); b[1] = B
    return LinearGaussianSSabcd(a, b, C, D)
end

# m-d state, univariate data
function LinearGaussianSSabcd(A::Matrix, B::Matrix, C::Matrix, D::Real)
    d = Array(Float64, 1, 1); d[1] = D
    return LinearGaussianSSabcd(A, B, C, d)
end

## ------------ ##
#- type: KFstep -#
## ------------ ##

immutable KFstep
    x_p::Vector{Float64}  # predicted state
    P_p::Matrix{Float64}  # predicted cov
    x_f::Vector{Float64}  # updated (filtered) state
    P_f::Matrix{Float64}  # updated (filtered) cov
    y_t::Vector{Float64}  # data for this step
    ll::Float64           # log-likelihood of this step
end

## ------------------ ##
#- Kalman filter code -#
## ------------------ ##

# helper functions
_get_x0_P0(x::MvNormal) = (mean(x), cov(x))
_get_x0_P0(x::Vector) = (x, eye(length(x)))

function get_T_ny_fixy(m::LinearGaussianSSabcd, y)
    # y should have observations as columns
    T = size(y, 2)
    ny = size(y, 1)
    ny_model = size(m.C[1], 1)

    if ny_model != ny
        T, ny = ny, T
        y = y'
    end

    return y, T, ny
end

function kfilter_step(m::LinearGaussianSSabcd,
                      x::Vector,
                      P::Matrix,
                      t::Int,
                      y_t)
    # unpack model parameters
    A, B, C, D = m.A, m.B, m.C, m.D

    # prediction
    x_p = A[t]*x                      # predict state
    P_p = A[t]*P*A[t]' + B[t]B[t]'    # predicted covariance

    # observation
    yhat = C[t]*x                     # predicted obs
    y_tilde = y_t - yhat              # innovation
    V = C[t]*P_p*C[t]' + D[t]*D[t]'   # innovation cov

    # update
    V_inv = inv(V)                    # don't repeat computing V^{-1}
    PT_CT_Vinv = P_p'*C[t]*V_inv      # useful object
    x_f = x_p + PT_CT_Vinv*y_tilde    # updated state
    P_f = P_p - PT_CT_Vinv*C[t]*P_p   # Update covariance

    # compute log-likelihood for this step
    ll = logpdf(MvNormal(yhat, V), y_t)

    # _p variables are for prediction, _f for filtered
    return KFstep(x_p, P_p, x_f, P_f, ll)
end


function kfilter(m::LinearGaussianSSabcd, y, x0)
    # Make sure y has obs in columns. Compute number obs and dim(y)
    y, T, ny = _get_T_ny_fixy(m, y)

    # compute the initial state and covariance
    x, P = _get_x0_P0(x0)

    # make sure dimensionality of initial state is correct
    nx = length(x0)
    @assert nx = size(m.A[1], 1) "Initial distribution incompatible with system"

    # allocate space for filter output
    filtered = Array(KFstep, T)

    # fill first observation
    filtered[1] = kfilter_step(m, x, P, 1, y[:, 1])

    # run the filter for the rest of the periods
    for t=2:T
        y_t = [:, t]  # this period's observation
        x, P = filtered[t-1].x_f, filtered[t-1].P_f  # Prev updated state/cov
        filitered[t] = kfilter_step(m, x, P, t, y_t)
    end
    filtered
end

loglik(s::Vector{KFstep}) = sum([x.ll for x in s])


function test_kfilter()
    # read in python data
    y = readdlm("/Users/grantlyon/Spencer/temp/y.csv", ',')
    ll_py = -223.02877677509935

    N = 2

    # generate state space matrices
    F = eye(N)
    V = .25 .* eye(N)
    G = eye(N)
    W = 0.25 .* eye(N)

    A, B, C, D = F, sqrtm(V), G, sqrtm(W)
    lgs = LinearGaussianSSabcd(A, B, C, D)

    x0_dist = MvNormal(ones(N), 0.1 .* eye(N))

    # run once to warm up
    ll_k = loglik(kfilter(lgs, y, x0_dist))

    # now time it
    tic()
    ll_k = loglik(kfilter(lgs, y, x0_dist))
    println("Execution time for kfilter: $(toq())")

    err = abs(ll_k - ll_py)
    pct_err = (err / abs(ll_pt)) * 100
    println("Absolute difference between Py and jl: $err ($pct_err %)")
end




