#************************************************************
# The main entrance for optimization
#************************************************************
function optimize{T}(p::OptimizationProblem, x::Vector{T};
    direction::Symbol = :gradient,
    stepsize::Symbol = :wolfe,
    maxiter::Integer = 500,
    xtol::Real = 1e-15,
    ftol::Real = 1e-5,
    gtol::Real = 1e-5,
    verbose::Bool = false,
    store_x::Bool = false,  # keep records of x
    store_f::Bool = true,   # keep records of function values
    store_g::Bool = false,  # keep records of gradients
    store_d::Bool = false,  # keep records of moving direction
    store_t::Bool = false,  # keep records of step size

    wolfe_opt = {},         # only used in wolfe linesearch
    backtrack_opt = {},     # only used in backtrack linesearch
    )

    #----------------------------------------
    # preparation
    #----------------------------------------
    dim = dimension(p)
    ws = WorkingSet{T}(
        x, zeros(T, dim),
        zeros(T, dim), zeros(dim), 
        objval(p, x), convert(T,0),
        zeros(T, dim),
        convert(T, 0), 0,
        store_f ? zeros(T, maxiter+1) : T[],
        store_x ? zeros(T, dim, maxiter+1) : Array(T, 0, 0),
        store_g ? zeros(T, dim, maxiter) : Array(T, 0, 0),
        store_d ? zeros(T, dim, maxiter) : Array(T, 0, 0),
        store_t ? zeros(T, maxiter) : T[]
    )
    if store_x; ws.x_all[:,1] = x; end
    if store_f; ws.f_all[1] = ws.f; end

    if direction == :gradient
        calcdir!() = ws.d = -ws.g
    elseif direction == :newton
        calcdir!() = ws.d = solve_hessian(p, ws.x_prev, -ws.g)
    else
        error("Unknown option <$direction> for direction calculating")
    end

    if stepsize == :exact
        linesearcher = linesearch
    elseif stepsize == :wolfe
        linesearcher = wolfe_linesearcher(; wolfe_opt...)
    elseif stepsize == :backtrack
        linesearcher = backtrack_linesearcher(; backtrack_opt...)
    else
        error("Unknown linesearch method: <$linesearch>")
    end
    
    #----------------------------------------
    # loop
    #----------------------------------------
    while ws.iter < maxiter
        ws.iter += 1
        ws.x_prev = ws.x
        ws.f_prev = ws.f

        # compute gradient
        ws.g_prev = ws.g
        ws.g = gradient(p, ws.x_prev)

        # compute direction
        calcdir!()
        
        # linesearch
        ws.t = linesearcher(p, ws.x_prev, ws.d)

        # move
        ws.x = ws.x_prev + ws.t*ws.d
        ws.f = objval(p, ws.x)

        # traces
        if verbose
            @printf " %4d: %.6f (%f)\n" ws.iter ws.f ws.f_prev-ws.f
        end

        # recording for debugging / visualization
        if store_x; ws.x_all[:,ws.iter+1] = ws.x; end
        if store_g; ws.g_all[:,ws.iter] = ws.g; end
        if store_d; ws.d_all[:,ws.iter] = ws.d; end
        if store_t; ws.t_all[ws.iter] = ws.t; end
        if store_f; ws.f_all[ws.iter+1] = ws.f; end

        # stop condition
        if abs(ws.f_prev - ws.f) < ftol; break; end
        if norm(ws.x-ws.x_prev) < xtol; break; end
        if ws.iter > 1
            if norm(ws.g) < gtol; break; end
        end
    end

    return ws
end

