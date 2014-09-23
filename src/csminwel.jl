#=

@author : Jonathan Payne <jep459@nyu.edu>,
@author : Spencer Lyon <spencer.lyon@nyu.edu>

@date: 2014-Sep-22

References
----------

Simple port of the original Matlab files csminwl.m, numgrad.m, bfgsi.m
written by Chris Sims.

=#

using Calculus: gradient  # for numerical derivatives
using Optim: update!, OptimizationTrace

macro csmtrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["x"] = copy(x)
            end
            grnorm = NaN
            update!(tr,
                    iteration,
                    f_x,
                    grnorm,
                    dt,
                    store_trace,
                    show_trace)
        end
    end
end


@doc* md"""

This routine implements Chris Sims' `csminwel` algorithm found
[here](http://sims.princeton.edu/yftp/optimize/).

This is a port of the MATLAB version of that function.

### Parameters

* `f::Function` : The objective function
* `grad::Function` : The gradient of the objective function
* `x0::Array`: The starting guess for the optimizer

#### Optional Arguments

* `H0::Matrix`: An initial guess for the Hessian matrix -- must be
  positive definite. If none is given, then a scaled down identity
  matrix is used.
* `args...`:  Other positional arguments to be passed to `f` on each
  function call

#### Keyword Arguments (optional)

* `ftol::{T<:Real}=1e-14`: Threshold for convergence in terms of change
  in function value across iterations.
* `nit::Int=100`: Maximum number of iterations
* `verbose::Bool=false`: Whether or not to print status messages as the
  algorithm progresses
* `io::IO=STDOUT`: The `IO` object to print messages to. Defaults to
  STDOUT, which is typically the REPL if Julia is run interactively.
* `kwargs...`: Other keyword arguments to be passed to `f` on each
  function call


#### Example

See the file `examples/csminwel.jl` for an example of usage
""" ->
function csminwel(fcn::Function, grad::Function, x0::Vector,
                  H0::Matrix=0.5.*eye(length(x0)), args...;
                  ftol::Float64=1e-14, nit::Int=100, verbose::Bool=false,
                  io::IO=STDOUT, kwargs...)
    nx = size(x0, 1)
    no = size(x0, 2)
    nx = max(nx , no)
    done = false
    itct = 0
    fcount = 0
    snit = 100

    f0 = fcn(x0, args...; kwargs...)

    if f0 > 1e50
        throw(ArgumentError("Bad initial parameter."))
    end

    g, badg = grad(x0)

    retcode3 = 101
    x = x0
    f = f0
    H = H0
    cliff = 0

    # set up return variables so they are available outside while loop
    fh = copy(f0)
    xh = copy(x0)
    gh = copy(x0)
    retcodeh = 1000

    while !done
        verbose && println(io, "-----------------")
        verbose && println(io, "-----------------")
        verbose && println(io, "f at the beginning of new iteration: $f")
        #-----------Comment out this line if the x vector is long-------------
        verbose && println(io, "x = $x")

        itct += 1
        f1, x1, fc, retcode1 = csminit(fcn, x, f, g, badg, H, args...;
                                       verbose=verbose, io=io, kwargs...)
        fcount += fc

        if retcode1 != 1
            if retcode1 == 2 || retcode1 == 4
                wall1, badg1 = true, true
            else
                g1, badg1 = grad(x1)
                wall1 = badg1
            end

            if wall1 && (length(H) > 1)
                # Bad gradient or back and forth on step length.  Possibly at
                # cliff edge. Try perturbing search direction if problem not
                # 1D

                Hcliff = H + diagm(diag(H).*rand(nx))
                verbose && println(io, "Cliff.  Perturbing search direction.")
                f2, x2, fc, retcode2 = csminit(fcn, x, f, g, badg, Hcliff,
                                               args...; verbose=verbose, io=io,
                                               kwargs...)
                fcount = fcount+fc; # put by Jinill
                if f2 < f
                    if retcode2==2 || retcode2==4
                        wall2 = 1; badg2 = 1
                    else
                        g2, badg2 = grad(x2)
                        wall2 = badg2
                        badg2

                    end
                    if wall2
                        verbose && println(io, "Cliff again.  Try traversing")
                        if norm(x2-x1) < 1e-13
                            f3 = f; x3 = x; badg3 = 1; retcode3 = 101
                        else
                            gcliff = ( (f2-f1) / ((norm(x2-x1))^2) )*(x2-x1)
                            if (size(x0 , 2)>1)
                                gcliff = gcliff'
                            end
                            f3, x3, fc, retcode3 = csminit(fcn, x, f, gcliff,
                                                           0, eye(nx),
                                                           args...;
                                                           verbose=verbose,
                                                           io=io, kwargs...)
                            fcount = fcount+fc
                            if retcode3==2 || retcode3==4
                                wall3 = 1; badg3 = 1
                            else
                                g3, badg3 = grad(x3)
                                wall3 = badg3
                            end
                        end
                    else
                        f3 = f
                        x3 = x
                        badg3 = true
                        retcode3 = 101
                    end
                else
                    f3 = f
                    x3 = x
                    badg3 = true
                    retcode3 = 101
                end
            else
                # normal iteration, no walls, or else 1D, or else we're
                # finished here.
                f2, f3 = f, f
                badg2, badg3 = true, true
                retcode2, retcode3 = 101, 101
            end
        else
            f1, f2, f3 = f, f, f
            retcode2, retcode3 = retcode1, retcode1
        end

        # how to pick gh and xh
        if f3 < f - ftol && badg3==0
            ih = 3
            fh = f3
            xh = x3
            gh = g3
            badgh = badg3
            retcodeh = retcode3
        elseif f2 < f - ftol && badg2==0
            ih = 2
            fh = f2
            xh = x2
            gh = g2
            badgh = badg2
            retcodeh = retcode2
        elseif f1 < f - ftol && badg1==0
            ih = 1
            fh = f1
            xh = x1
            gh = g1
            badgh = badg1
            retcodeh = retcode1
        else
            fh, ih = findmin([f1 , f2 , f3])
            verbose && println(io, "ih = $ih")
            if ih == 1
                xh = x1
                retcodeh = retcode1
            elseif ih == 2
                xh = x2
                retcodeh = retcode2
            elseif ih == 3
                xh = h3
                retcodeh = retcode3
            end

            if isdefined(:gh)
                nogh = isempty(gh)
            else
                nogh = true
            end

            if nogh
                gh, badgh = grad(xh)
            end
            badgh = true
        end

        stuck = (abs(fh-f) < ftol)
        if !badg && !badgh && !stuck
            H = bfgsi(H , gh-g , xh-x)
        end
        verbose && println(io, "----")
        verbose && println(io, "Improvement on iteration $itct = $(f-fh)")

        if itct > nit
            verbose && println(io, "iteration count termination")
            done = true
        elseif stuck
            verbose && println(io, "improvement < ftol termination")
            done = true
        end

        rc = retcodeh
        if rc == 1
            verbose && println(io, "zero gradient")
        elseif rc == 6
            verbose && println(io, "smallest step still improving too slow,",
                                    " reversed gradient")
        elseif rc == 5
            verbose && println(io, "largest step still improving too fast")
        elseif rc == 4 || rc==2
            verbose && println(io, "back and forth on step length never",
                                   "finished")
        elseif rc == 3
            verbose && println(io, "smallest step still improving too slow")
        elseif rc == 7
            verbose && println(io, "warning: possible inaccuracy in H matrix")
        end

        # update before next iteration
        f = fh
        x = xh
        g = gh
        badg = badgh
    end

    return fh, xh, gh, H, itct, fcount, retcodeh
end

@doc md"""
Version of `csminwel` that will use finite differencing methods to
approximate the gradient numerically. This is convenient for cases where
you cannot supply an analytical derivative, but it is not as robust as
using the true derivative.
""" ->
function csminwel(fcn::Function, x0::Vector, H0::Matrix=0.5.*eye(length(x0)),
                  args...; ftol::Float64=1e-14, nit::Int=100,
                  verbose::Bool=false, io::IO=STDOUT, kwargs...)
    grad(x) = numgrad(fcn, x, args...; kwargs...)
    csminwel(fcn, grad, x0, H0, args...; ftol=ftol, nit=nit, verbose=verbose,
             io=io, kwargs...)
end



function numgrad(fcn, x, args...; kwargs...)
    g = gradient(a -> fcn(a, args...; kwargs...), x)
    bads = abs(g) .>= 1e15
    g[bads] = 0.0
    return g, any(bads)
end

# SL: This function worked for my example (2014-09-23 09:08:52)
function csminit(fcn, x0, f0, g0, badg, H0, args...;
                 verbose::Bool=false, io::IO=STDOUT, kwargs...)
    angle = .005

    #(0<THETA<.5) THETA near .5 makes long line searches, possibly fewer
    # iterations.
    theta = .3
    fchange = 1000
    minlamb = 1e-9
    mindfac = .01
    fcount = 0
    lambda = 1.0
    xhat = x0
    f = f0
    fhat = f0
    g = g0
    gnorm = norm(g)

    if gnorm < 1e-12 && !badg
        # gradient convergence
        retcode = 1
        dxnorm = 0.0
    else
        # with badg true, we don't try to match rate of improvement to
        # directional derivative.  We're satisfied just to get some
        # improvement in f.
        dx = vec(-H0*g)
        dxnorm = norm(dx)

        if dxnorm > 1e12
            verbose && println(io, "Near singular H problem.")
            dx = dx * fchange / dxnorm
        end

        dfhat = dot(dx, g0)

        if !badg
            # test for alignment of dx with gradient and fix if necessary
            a = -dfhat / (gnorm*dxnorm)

            if a < angle
                dx -= (angle*dxnorm/gnorm + dfhat/(gnorm*gnorm)) * g
                dx *= dxnorm/norm(dx)
                dfhat = dot(dx, g)
                verbose && println(io, "Correct for low angle $a")
            end
        end

        verbose && println(io, "Predicted Improvement: $(-dfhat/2)")

        # Have OK dx, now adjust length of step (lambda) until min and
        # max improvement rate criteria are met.
        done = false
        fact = 3.0
        shrink = true
        lambda_min = 0.0
        lambda_max = Inf
        lambda_peak = 0.0
        f_peak = f0
        lambda_hat = 0.0

        while !done
            if size(x0, 2) > 1
                dxtest = x0 + dx' * lambda
            else
                dxtest = x0 + dx * lambda
            end

            f = fcn(dxtest, args...; kwargs...)
            verbose && println(io, "lambda = $lambda; f = $f")

            if f < fhat
                fhat = f
                xhat = dxtest
                lambdahat = lambda
            end

            fcount += 1
            shrink_signal = (!badg &&
                             (f0-f < maximum([-theta*dfhat*lambda 0]))) ||
                            (badg && (f0-f) < 0)

            grow_signal = !badg && (lambda > 0)  &&
                          (f0-f > -(1-theta)*dfhat*lambda)

            if shrink_signal && ((lambda > lambda_peak) || lambda < 0 )
                if (lambda > 0) && ((!shrink) || (lambda/fact <= lambda_peak))
                    shrink = true
                    fact = fact^.6
                    while lambda / fact <= lambda_peak
                        fact = fact^.6
                    end

                    if abs(fact - 1.0) < mindfac
                        if abs(lambda) < 4
                            retcode = 2
                        else
                            retcode = 7
                        end

                        done = true
                    end
                end

                if lambda < lambda_max && lambda > lambda_peak
                    lambda_max = lambda
                end

                lambda /= fact
                if abs(lambda) < minlamb
                    if (lambda > 0) && (f0 <= fhat)
                        lambda = -lambda*fact^6
                    else
                        if lambda < 0
                            retcode = 6
                        else
                            retcode = 3
                        end
                        done = true
                    end
                end


            elseif (grow_signal && lambda > 0) || (shrink_signal &&
                                    ((lambda <= lambda_peak) && (lambda > 0)))
                if shrink
                    shrink = false
                    fact = fact^.6
                    if abs(fact - 1) < mindfac
                        if abs(lambda) < 4
                            retcode = 4
                        else
                            retcode = 7
                        end
                        done = true
                    end
                end

                if f < f_peak && lambda > 0
                    f_peak = f
                    lambda_peak = lambda
                    if lambda_max <= lambda_peak
                        lambda_max = lambda_peak * fact^2
                    end
                end

                lambda *= fact
                if abs(lambda) > 1e20
                    retcode = 5
                    done = true
                end
            else
                done = true
                if fact < 1.2
                    retcode = 7
                else
                    retcode = 0
                end
            end
        end
    end

    verbose && println(io, "Norm of dx $dxnorm")
    return fhat, xhat, fcount, retcode
end


function bfgsi(H0, dg, dx; verbose::Bool=false, io::IO=STDOUT)
    # H = bfgsi(H0,dg,dx)
    # dg is previous change in gradient; dx is previous change in x;
    # 6/8/93 version that updates inverse hessian instead of hessian
    # itself.
    # Copyright by Christopher Sims 1996.  This material may be freely
    # reproduced and modified.
    if size(dg, 2) > 1
        dg = dg'
    end

    if size(dx, 2) > 1
        dx = dx'
    end

    Hdg = H0*dg
    dgdx = dot(dx, dg)

    if abs(dgdx) > 1e-12
        # SL: (1+(dg'*Hdg)/dgdx) this is a (1, 1) 2d array, but we should
        #     be treating it is a scalar. That's why I have .* after it not *
        H = H0 + (1+(dg'*Hdg)/dgdx).*(dx*dx')/dgdx - (dx*Hdg'+Hdg*dx')/dgdx
    else
        verbose && println(io, "bfgs update failed")
        verbose && println(io, "|dg| = $(norm(dg)), |dx| = $(norm(dx))")
        verbose && println(io, "dg'dx = $dgdx")
        verbose && println(io, "|H*dg| = $(norm(Hdg))")
        H = H0
    end
    return H
end


