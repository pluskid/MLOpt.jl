# A simple example that demonstrate
# gradient descent with exact line
# search on simple quadratic function

using MLOpt

# simple 2-d quadratic function
type SimpleQuadratic <: OptimizationProblem
    q::Vector{Float64} # diagonal of the Hessian
end

function MLOpt.dimension(p::SimpleQuadratic)
    length(p.q)
end

function MLOpt.objval(p::SimpleQuadratic, x::Vector{Float64})
    0.5*dot(x.*p.q, x)
end

function MLOpt.gradient(p::SimpleQuadratic, x::Vector{Float64})
    p.q.*x;
end

function MLOpt.linesearch(p::SimpleQuadratic, x::Vector{Float64}, d::Vector{Float64})
    dot(d,d) / dot(d.*p.q, d)
end


problem = SimpleQuadratic([1.0,200.0])
#problem = SimpleQuadratic([1.0,2.0])
ret = optimize(problem, [100.0, 4.0]; 
        stepsize=:exact, verbose=true,
        xtol=1e-3, store_x=true)


import PyPlot.plt

plt.figure(figsize=(12,5))

x = [-5:0.05:100]
y = [-5:0.05:5]
Z = (x.*x*problem.q[1])' .+ y.*y*problem.q[2]
plt.contour(x, y, Z)

plt.plot(ret.x_all[1,:]', ret.x_all[2,:]', color="green", linestyle="solid", marker="o")
fn = @sprintf "gd-quadratic-%d-%d" problem.q[1] problem.q[2]
plt.savefig("fig/$(fn).svg", transparent=true, bbox_inches="tight", pad_inches=0)
plt.savefig("fig/$(fn).pdf", transparent=true, bbox_inches="tight", pad_inches=0)

