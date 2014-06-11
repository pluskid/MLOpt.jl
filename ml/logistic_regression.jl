# Binary Logistic Regression
type LogisticRegression{T} <: MLModel
    X   ::Matrix{T}
    Y   ::Vector{T}
    lam ::T
end

function dimension{T}(p::LogisticRegression{T})
    size(p.X, 2)
end

function objval{T}(p::LogisticRegression{T}, w::Vector{T})
    -loglikelihood(p, w) + 0.5*p.lam*dot(w,w)
end

function gradient{T}(p::LogisticRegression{T}, w::Vector{T})
    expval = exp(-p.Y .* (p.X * w))
    vec(sum(- expval ./ (1+expval) .* p.Y .* p.X, 1)' + p.lam * w)
end

function solve_hessian{T}(p::LogisticRegression{T}, w::Vector{T}, d::Vector{T})
    expval = exp(-p.Y .* (p.X * w))
    weight = 1 ./ (expval + 1./expval + 2)
    hessian = (weight .* p.X)' * p.X + p.lam * eye(size(p.X, 2))
    hessian \ d
end

function classify{T}(p::LogisticRegression{T}, w::Vector{T}, X::Matrix{T})
    sign(X * w)
end

function loglikelihood{T}(p::LogisticRegression{T}, w::Vector{T})
    sum(log(cond_prob(p, w)))
end
function cond_prob{T}(p::LogisticRegression{T}, w::Vector{T})
    1 ./ (1 + exp(-p.Y .* (p.X * w)))
end
