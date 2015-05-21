type Cooccurence{Ti,Tj<:Int, T}
    i::Ti
    j::Tj
    v::T
end

CoVector(n::Int) = Array(Cooccurence, n)
function CoVector(comatrix::SparseMatrixCSC)
    aa = findnz(comatrix)
    n = length(aa[1])
    a = CoVector(n)
    @inbounds for i = 1:n
        a[i] = Cooccurence(aa[1][i], aa[2][i], aa[3][i])
    end
    a
end

abstract Solver

type Adagrad <: Solver
    niter::Int
    lrate::Float64
end

Adagrad(niter) = Adagrad(niter, 0.05)
Adagrad(niter, lrate) = Adagrad(niter, lrate)

# GloVe model
# TODO: combine bias with weight matrix?
type Model{T}
    W_main::Matrix{T}
    W_ctx::Matrix{T}
    b_main::Vector{T}
    b_ctx::Vector{T}
    W_main_grad::Matrix{T}
    W_ctx_grad::Matrix{T}
    b_main_grad::Vector{T}
    b_ctx_grad::Vector{T}
    covec::Vector{Cooccurence}
end

# Each vocab word in associated with a word vector and a context vector.
# The paper initializes the weights to values [-0.5, 0.5] / vecsize+1 and
# the gradients to 1.0.
#
# The +1 term is for the bias.
function Model(comatrix; vecsize=100)
    vs = size(comatrix, 1)
    Model(
        (rand(vs, vecsize) - 0.5) / (vecsize + 1),
        (rand(vs, vecsize) - 0.5) / (vecsize + 1),
        (rand(vs) - 0.5) / (vecsize + 1),
        (rand(vs) - 0.5) / (vecsize + 1),
        ones(vs, vecsize),
        ones(vs, vecsize),
        ones(vs),
        ones(vs),
        CoVector(comatrix),
    )
end

function train!(m::Model, s::Adagrad; xmax=100, alpha=0.75)
    J = 0.0

    shuffle!(m.covec)
    for n=1:s.niter
        # shuffle indices
        @inbounds for i = 1:length(m.covec)
            co = m.covec[i]

            # locations
            l1 = co.i # main
            l2 = co.j # context

            diff = m.W_main[l1,:] * m.W_ctx[l2,:]' + m.b_main[l1] + m.b_ctx[l2] - log(co.v)
            fdiff = ifelse(co.v < xmax, (co.v / xmax) ^ alpha, 1.0) * diff
            J += 0.5 * fdiff * diff

            fdiff *= s.lrate
            grad_main = fdiff * m.W_ctx[l2, :]
            grad_ctx = fdiff * m.W_main[l1, :] 

            # Adaptive learning
            m.W_main[l1, :] -= grad_main ./ sqrt(m.W_main_grad[l1, :])
            m.W_ctx[l2, :] -= grad_ctx ./ sqrt(m.W_ctx_grad[l2, :])
            m.b_main[l1, :] -= fdiff ./ sqrt(m.b_main_grad[l1, :])
            m.b_ctx[l2, :] -= fdiff ./ sqrt(m.b_ctx_grad[l2, :])

            # Gradients
            fdiff *= fdiff
            m.W_main_grad[l1, :] += grad_main .^ 2
            m.W_ctx_grad[l2, :] += grad_ctx .^ 2
            m.b_main_grad[l1, :] += fdiff
            m.b_ctx_grad[l2, :] += fdiff

        end

        if n % 10 == 0
            println("iteration $n, cost $J")
        end
    end

    #= vecsize = size(m.W_main, 2) =#
    # Average the main and context vectors
    #= m.W_main[1:end, :] += m.W_ctx[1:end, :] =#
    #= m.W_main /= vecsize =#
end

# Average the main and context matrices/bias vectors. 
function combine(m::Model)
    vecsize = size(m.W_main, 2)
    M = m.W_main + m.W_ctx .+ (m.b_main / vecsize) .+ (m.b_ctx / vecsize)
    M /= (vecsize + 1)
    M
end
