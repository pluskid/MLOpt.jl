# simple demonstration of logistic regression on USPS
# handwritten digit recognition using gradient descent
# with wolfe's condition for line search

using MAT
using MLOpt

function runtest(X, Y, Xtt, Ytt, lam=0.0)
    problem = LogisticRegression{Float64}(X, Y, lam)
    ret = optimize(problem, rand(dimension(problem)); verbose=true)
    w = ret.x

    Yhat = classify(problem, w, Xtt)
    accuracy = float(sum(Yhat .== Ytt)) / length(Ytt)
    @printf "accuracy = %.4f\n" 100*accuracy
    return ret
end

# load data
matfile = matopen("data/usps.mat")
X = read(matfile, "data")'
Y = vec(read(matfile, "label"))

# classify digit '1' and '2'
mask = (Y .== 1) | (Y .== 2)
X = X[mask,:]
Y = Y[mask]
Y[Y .== 2] = -1
Y[Y .== 1] = 1
Y = convert(Vector{Float64}, Y)

# partition train-test
N = length(Y)
Ntr = round(N/2)
Ntt = N-Ntr
rp = randperm(N)

Xtr = X[rp[1:Ntr],:]
Ytr = Y[rp[1:Ntr]]
Xtt = X[rp[Ntr+1:end],:]
Ytt = Y[rp[Ntr+1:end]]

ret = runtest(Xtr, Ytr, Xtt, Ytt, 0.0)

import PyPlot.plt
plt.figure(figsize=(8,3))
plt.semilogy([1:ret.iter+1], ret.f_all[1:ret.iter+1], color="green", linestyle="solid", marker="*")
plt.xlabel("Iteration")
plt.ylabel("Objective Function")
plt.tight_layout()
fn = "gd-logistic-regression"
plt.savefig("fig/$(fn).svg", transparent=true, bbox_inches="tight", pad_inches=0)
plt.savefig("fig/$(fn).pdf", transparent=true, bbox_inches="tight", pad_inches=0)
