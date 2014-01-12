function wolfe_linesearcher(; c1=0.01, c2=0.9, t0=1, tol=0.001)
    function linesearch{T}(p::OptimizationProblem, x::Vector{T}, d::Vector{T})
        low = convert(T, 0)
        high = inf(T)

        t = convert(T, t0)
        fx = objval(p, x)
        dfx = gradient(p, x)

        while true
            if objval(p, x + t*d) > fx + c1*t*dot(dfx, d)
                high = t
                t = (low + high) / 2
            elseif dot(gradient(p, x+t*d), d) < c2*dot(dfx, d)
                low = t
                if isinf(high)
                    t = 2*low
                else
                    t = (low+high)/2
                end
            else
                break
            end
            if high-low < tol
                break
            end
        end
        return t
    end

    return linesearch
end
