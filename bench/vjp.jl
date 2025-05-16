using LinearAlgebra, LinearSolve, SciMLOperators
using Enzyme

function f!(out1, out2, in1, in2, A, B)
    out1 .= A * in1
    out2 .= B * in2
end

function vjp(din, dout, p, t)
    (in1, in2, out1, out2, A, B) = p
    dout1, dout2 = dout[1:3], dout[4:6]
    din1, din2 = zeros(3), zeros(3)
    autodiff(Reverse, f!, Const, Duplicated(out1, dout1), Duplicated(out2, dout2),
        Duplicated(in1, din1), Duplicated(in2, din2), Const(A), Const(B))
    din[1:3] .= din1
    din[4:6] .= din2
    din
end

in1, in2, out1, out2 = rand(3), rand(3), rand(3), rand(3)
A = Symmetric(rand(3, 3) + 2I)
B = Symmetric(rand(3, 3) + 2I)
op = FunctionOperator(vjp, zeros(6); p=(in1, in2, out1, out2, A, B))
b = rand(6)
prob = LinearProblem(op, b)
sol = solve(prob)
[A \ b[1:3]; B \ b[4:6]]

(; H, S, O, ERI, F, P, C, ε) = cache
proto = zeros(length(C) * 2)
f1 = zeros(size(C))
f2 = zeros(size(C))
e = diagm(ε)
op = FunctionOperator(vjp, proto; p=(C, e, f1, f2, H, S, ERI, O))
b = rand(length(C) * 2)
prob = LinearProblem(op, b)
sol = solve(prob)
df1
