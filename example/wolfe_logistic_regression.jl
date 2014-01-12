# simple demonstration of logistic regression on USPS
# handwritten digit recognition using gradient descent
# with wolfe's condition for line search

using MAT
using MLOpt

function runtest(X, Y, lam=0.0)
    problem = LogisticRegression{Float64}(X, Y, lam)
    ret = optimize(problem, rand(dimension(problem)); verbose=true)
    w = ret.x

    Yhat = classify(problem, w, X)
    accuracy = float(sum(Yhat == Y)) / length(Y)
    println(accuracy)
end

matfile = matopen("data/usps.mat")
X = read(matfile, "data")'
Y = convert(Vector{Float64}, vec(read(matfile, "label")))

runtest(X, Y)
