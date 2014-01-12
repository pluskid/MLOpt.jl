# Binary Logistic Regression
type LogisticRegression{T} <: MLModel
    X   ::Matrix{T}
    Y   ::Vector{Integer}
    lam ::T
end

function dimension{T}(p::LogisticRegression{T})
    size(X, 1)
end

function objval{T}(p::LogisticRegression{T}, w::Vector{T})
    -loglikelihood(p, w) + 0.5*p.lam*dot(w,w)
end

function gradient{T}(p::LogisticRegression{T}, w::Vector{T})
    expval = exp(-p.Y .* dot(p.X, w))
    sum(- expval ./ (1+expval) .* p.Y .* p.X, 2) + p.lam * w
end

function classify{T}(p::LogisticRegression{T}, w::Vector{T}, X::Matrix{T})
    sign(dot(X, w))
end

function loglikelihood{T}(p::LogisticRegression{T}, w::Vector{T})
    sum(log(cond_prob(p, w)))
end
function cond_prob{T}(p::LogisticRegression{T}, w::Vector{T})
    1 ./ (1 + exp(-p.Y .* dot(p.X, w)))
end
