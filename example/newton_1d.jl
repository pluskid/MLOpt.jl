# A simple example demonstrating the convergence and
# convergence rate of Newton's method

using MLOpt

# f(x) = log(e^x + e^-x)
type DemoProblem <: OptimizationProblem
end

function MLOpt.dimension(p::DemoProblem)
    1
end

function MLOpt.objval(p::DemoProblem, x::Vector{Float64})
	x = x[1]
    log(exp(x) + exp(-x))
end

function MLOpt.gradient(p::DemoProblem, x::Vector{Float64})
	x = x[1]
    a = exp(x)
    b = exp(-x)
    [(a-b) / (a+b)]
end

function MLOpt.linesearch(p::DemoProblem, x::Vector{Float64}, d::Vector{Float64})
	1 # don't do line search
end

function MLOpt.solve_hessian(p::DemoProblem, x::Vector{Float64}, d::Vector{Float64})
	x = x[1]
	a = exp(x)
	b = exp(-x)
	h = 4 / (a+b)^2
	[d / h]
end


problem = DemoProblem()
init_points = [1.08, 1.09, 25.0]
algorithms = [("Newton", :newton, :exact),
			  ("Damped Newton", :newton, :backtrack),
			  ("Gradient Descent", :gradient, :exact)]
colors = ["green", "red", "black"]
msizes = [4, 8, 4]

import PyPlot.plt
plt.figure(figsize=(14,3))

for i = 1:length(init_points)
	init_x = [init_points[i]]
	plt.subplot(1,3,i)

	for j = 1:length(algorithms)
		algor = algorithms[j]
		ret = optimize(problem, init_x; direction=algor[2], stepsize=algor[3], 
			verbose=true, store_f=true, maxiter=25)
		f_all = ret.f_all[1:ret.iter]
		f_all[isnan(f_all) | isinf(f_all)] = 50
		plt.plot(1:ret.iter, f_all, color=colors[j], linestyle="solid", 
			marker="o", markersize=msizes[j], label=algor[1])
		plt.ylim(0.68, 1.5*maximum(ret.f_all))
	end
	plt.legend()
	plt.xlabel("Iteration")
	plt.ylabel("Function Value")
	plt.title(@sprintf "init = %.2f" init_x[1])
end

plt.savefig("fig/newton-1d.pdf", transparent=true, bbox_inches="tight", pad_inches=0)
plt.savefig("fig/newton-1d.svg", transparent=true, bbox_inches="tight", pad_inches=0)

# plot the function
plt.figure(figsize=(3, 4))
x = linspace(-15,15,100)
y = log(exp(x) + exp(-x))
plt.plot(x,y)
plt.title("log(exp(x)+exp(-x))")
plt.savefig("fig/1d-func.pdf", transparent=true, bbox_inches="tight", pad_inches=0)
plt.savefig("fig/1d-func.svg", transparent=true, bbox_inches="tight", pad_inches=0)
