# A demo for using projected gradient descent to solve
# the LASSO problem.
#
# NOTE: wavelet transformations from the package Wavelets.jl are used
# in this demo. At the time this demo is written, Wavelets.jl is a 
# standalone package. But it is not included in the standard julia
# package index, because there is a plan to incorporate this package
# into the larger julia/DSP package. Currently you could install this
# package by simply cloning the git repo to your local package storage
# (typically ~/.julia or ~/.julia/v0.3)
#

using MLOpt
using Images
using Wavelets

type DemoProblem <: OptimizationProblem
    img::Image{Float64,2,Array{Float64,2}}
    lambda::Float64
    wavelet::WaveletType

    img_fwt::Vector{Float64} # cached for computing gradient
end

function MLOpt.dimension(p::DemoProblem)
    2*length(p.img.data)
end

function MLOpt.objval(p::DemoProblem, x::Vector{Float64})
    rcv_img = recover_image(p, x)
    dif_img = rcv_img - p.img.data
    return sum(dif_img .* dif_img) + p.lambda * sum(x)
end

function MLOpt.gradient(p::DemoProblem, x::Vector{Float64})
    x_pos, x_neg = split_x(x)
    x_comb = x_pos + x_neg

    2([x_comb,x_comb] - p.img_fwt) + p.lambda * ones(MLOpt.dimension(p))
end

function MLOpt.linesearch(p::DemoProblem, x::Vector{Float64}, d::Vector{Float64})
    1 # constant step size
end

function MLOpt.projection(p::DemoProblem, x::Vector{Float64})
    max(x, 0) # projection to non-negative cone
end


# x is represented as x_pos - x_neg
function split_x(x::Vector{Float64})
    real_len = length(x)/2
    x_pos = x[1:real_len]
    x_neg = x[real_len+1:end]
    return (x_pos, x_neg)
end
function real_x(x::Vector{Float64})
    x_pos, x_neg = split_x(x)
    return x_pos-x_neg
end
function recover_image(p::DemoProblem, x::Vector{Float64})
    iwt(reshape(real_x(x), size(p.img.data)), p.wavelet)
end


#****************************************
# Prepare data & problem
#****************************************
lambda = 0.1
wavelet = POfilter("sym5")
img = float64sc(imread("data/lena512gray.bmp"))
img_fwt = reshape(fwt(img.data, wavelet), length(img.data))
img_fwt = [img_fwt, -img_fwt]

imwrite(img, "fig/pgd-lena-orig.jpg")

# add noise to the image
img.data += 0.1*randn(size(img.data))
imwrite(img, "fig/pgd-lena-noisy.jpg")

# initialize optimization problem
problem = DemoProblem(img, lambda, wavelet, img_fwt)
init_x = MLOpt.projection(problem, randn(dimension(problem)))

# run optimization
ret = optimize(problem, init_x; direction = :gradient, stepsize = :exact,
                                do_projection = true, stepsize_ap = :backtrack,
                                maxiter = 25, verbose = true)

# recover denoised image
rcv_img = recover_image(problem, ret.x)
imwrite(rcv_img', "fig/pgd-lena-rcv.jpg")
