using LinearAlgebra, LinearSolve, SciMLOperators
using Enzyme

function f!(out1, out2, in1, in2, A, B)
    out1 .= A * in1
    out2 .= B * in2
end

function vjp(din, dout, p, t)
    (in0, out, A, B) = p
    din .= 0.0
    doutclone = copy(dout)
    dout1, dout2 = (@view dout[1:3]), (@view dout[4:6])
    din1, din2 = (@view din[1:3]), (@view din[4:6])
    out1, out2 = (@view out[1:3]), (@view out[4:6])
    in1, in2 = (@view in0[1:3]), (@view in0[4:6])
    autodiff(Reverse, f!, Const, Duplicated(out1, dout1), Duplicated(out2, dout2),
        Duplicated(in1, din1), Duplicated(in2, din2), Const(A), Const(B))
    dout .= doutclone
    din
end

in0, out = rand(6), rand(6)
A = Symmetric(rand(3, 3) + 2I)
B = Symmetric(rand(3, 3) + 2I)
op = FunctionOperator(vjp, zeros(6); p = (in0, out, A, B))
b = rand(6)
prob = LinearProblem(op, b)
sol = solve(prob)
[A \ b[1:3]; B \ b[4:6]]
