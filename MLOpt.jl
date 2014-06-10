module MLOpt

include("opt/type.jl")

include("opt/optimize.jl")
include("opt/line_searcher.jl")

include("ml/base.jl")
include("ml/logistic_regression.jl")

export OptimizationProblem
export optimize
export dimension, objval, gradient, hessian!

export MLModel, LogisticRegression
export classify

end
