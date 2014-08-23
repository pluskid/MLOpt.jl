#************************************************************
# Abstract type that represent an optimization
# problem
#************************************************************
abstract OptimizationProblem

#------------------------------------------------------------
# A concrete optimization problem should implement the
# following API
#------------------------------------------------------------
function dimension(p::OptimizationProblem)
    error("return dimension of the variable")
end

function objval{T}(p::OptimizationProblem, x::Vector{T})
    error("return the objective function value at x")
end

function gradient{T}(p::OptimizationProblem, x::Vector{T})
    error("return the gradient at x")
end

# this function will only be called in 2nd order methods, it returns the solution
# of the equation: H(x) * z = d, where H(x) is the Hessian matrix at point x
function solve_hessian{T}(p::OptimizationProblem, x::Vector{T}, d::Vector{T})
    error("solve for hessian at x and store it in g")
end

# this function will only be called when exact linesearch is used
function linesearch{T}(p::OptimizationProblem, x::Vector{T}, d::Vector{T})
    error("compute exact line search at x in direction d")
end

# this function do necessary projection into the feasible set
# for constrained optimization
function projection{T}(p::OptimizationProblem, x::Vector{T})
    error("project x into the feasible region")
end


#************************************************************
# Working set, stores everything during optimization
#************************************************************
type WorkingSet{T}
    x             ::Vector{T}
    x_prev        ::Vector{T}
    g             ::Vector{T}
    g_prev        ::Vector{T}
    f             ::T
    f_prev        ::T
    d             ::Vector{T}
    t             ::T
    iter          ::Integer
    f_all         ::Vector{T}
    x_all         ::Matrix{T}
    g_all         ::Matrix{T}
    d_all         ::Matrix{T}
    t_all         ::Vector{T}
end
