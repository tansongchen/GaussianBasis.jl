using Enzyme, Enzyme.EnzymeRules
import Enzyme.EnzymeRules: forward, reverse, augmented_primal
using LinearAlgebra

function f(A, B)
    e, C = eigen(A, B)
    sum(e)
end

function augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(f)}, ::Type{<:Active},
    A::Duplicated, B::Duplicated)
    primal = f(A.val, B.val)
    return AugmentedReturn(primal, nothing, nothing)
end

function reverse(config::RevConfigWidth{1}, func::Const{typeof(f)}, dret::Active, tape,
    A, B)
    # do something to calculate the gradient
    return (nothing, nothing)
end

A = Symmetric([3. 2.; 2. 3.])
B = Symmetric([2. 1.; 1. 2.])
f(A, B)
dA = Symmetric(zeros(size(A)))
dB = Symmetric(zeros(size(B)))
autodiff(ReverseWithPrimal, f, Active, Duplicated(A, dA), Duplicated(B, dB))
