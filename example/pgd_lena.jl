# A demo for using projected gradient descent to solve
# the LASSO problem.
#
# Requires the following packages:
#
#   Pkg.add("Wavelets")
#   Pkg.add("Images")

using MLOpt
using Images
using Wavelets

type DemoProblem <: OptimizationProblem
    img::Image{Float64,2,Array{Float64,2}}
    lambda::Float64
    wavelet::DiscreteWavelet

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
    idwt(reshape(real_x(x), size(p.img.data)), p.wavelet)
end


#****************************************
# Prepare data & problem
#****************************************
lambda = 0.5
the_wavelet = wavelet(WT.sym5)
img = reinterpret(Float64, float64(imread("data/lena512gray.bmp")))

imwrite(img, "fig/pgd-lena-orig.jpg")

# add noise to the image
img.data += 0.2*randn(size(img.data))
imwrite(img, "fig/pgd-lena-noisy.jpg")

# initialize optimization problem
img_fwt = reshape(dwt(img.data, the_wavelet), length(img.data))
img_fwt = [img_fwt, -img_fwt]
problem = DemoProblem(img, lambda, the_wavelet, img_fwt)
init_x = MLOpt.projection(problem, randn(dimension(problem)))

# run optimization
ret = optimize(problem, init_x; direction = :gradient, stepsize = :exact,
                                do_projection = true, stepsize_ap = :backtrack,
                                maxiter = 25, verbose = true, ftol = 1e-10)

# recover denoised image
rcv_img = recover_image(problem, ret.x)
imwrite(rcv_img', "fig/pgd-lena-rcv.jpg")
