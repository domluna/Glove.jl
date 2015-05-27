# Cooccurence represents a co-occurence between two word ids.
# i is the id of the main word, j is the id of the context word.
# v is the co-occurence value between the two words.
immutable Cooccurence{Ti,Tj<:Int, T}
    i::Ti
    j::Tj
    v::T
end

# make_covector creates an Vector of Cooccurence's.
# In 0.4 this is most likely not required due to
# improvements to Sparse Linear Algebra.
function make_covector{T}(comatrix::SparseMatrixCSC{T})
    aa = findnz(comatrix)
    n = length(aa[1])
    a = Array(Cooccurence{Int, Int, T}, n)
    @inbounds for i = 1:n
        a[i] = Cooccurence(aa[1][i], aa[2][i], aa[3][i])
    end
    a
end

abstract Solver

# Adagrad is the method used for optimization in the paper. However,
# other methods may show better results.
#
# 
type Adagrad <: Solver
    niter::Int
    lrate::Float64
end

# 0.05 is the learning rate used in the paper
Adagrad(niter) = Adagrad(niter, 0.05)
Adagrad(niter, lrate) = Adagrad(niter, lrate)

# Glove model
type Model{T}
    W_main::Matrix{T}
    W_ctx::Matrix{T}
    b_main::Vector{T}
    b_ctx::Vector{T}
    W_main_grad::Matrix{T}
    W_ctx_grad::Matrix{T}
    b_main_grad::Vector{T}
    b_ctx_grad::Vector{T}
    covec::Vector{Cooccurence{Int, Int, T}}
end

# Each vocab word in associated with a word vector and a context vector.
# The paper initializes the weights to values [-0.5, 0.5] / vecsize+1 and
# the gradients to 1.0.
#
# The +1 term is for the bias.
function Model(comatrix; vecsize=100)
    vs = size(comatrix, 1)
    Model(
        (rand(vecsize, vs) - 0.5) / (vecsize + 1),
        (rand(vecsize, vs) - 0.5) / (vecsize + 1),
        (rand(vs) - 0.5) / (vecsize + 1),
        (rand(vs) - 0.5) / (vecsize + 1),
        ones(vecsize, vs),
        ones(vecsize, vs),
        ones(vs),
        ones(vs),
        make_covector(comatrix),
    )
end

function train!{T}(m::Model{T}, s::Adagrad; xmax::Int=100, alpha::T=0.75, verbose::Bool=false)
    J = 0.0

    shuffle!(m.covec)

    vecsize = size(m.W_main, 1)
    S = eltype(m.b_main)
    vm = zeros(S, vecsize)
    vc = zeros(S, vecsize)
    grad_main = zeros(S, vecsize)
    grad_ctx = zeros(S, vecsize)

    for n = 1:s.niter
        # shuffle indices
        for i = 1:length(m.covec)
            @inbounds l1 = m.covec[i].i # main index
            @inbounds l2 = m.covec[i].j # context index
            @inbounds v = m.covec[i].v

            @inbounds for j = 1:vecsize
                vm[j] = m.W_main[j, l1]
                vc[j] = m.W_ctx[j, l2]
            end

            diff = dot(vec(vm), vec(vc)) + m.b_main[l1] + m.b_ctx[l2] - log(v)
            fdiff = ifelse(v < xmax, (v / xmax) ^ alpha, 1.0) * diff
            J += 0.5 * fdiff * diff

            fdiff *= s.lrate
            @inbounds for j = 1:vecsize
                grad_main[j] = fdiff * m.W_ctx[j, l2]
                grad_ctx[j] = fdiff * m.W_main[j, l1]
            end

            # Adaptive learning
            @inbounds for j = 1:vecsize
                m.W_main[j, l1] -= grad_main[j] / sqrt(m.W_main_grad[j, l1])
                m.W_ctx[j, l2] -= grad_ctx[j] / sqrt(m.W_ctx_grad[j, l2])
            end
            m.b_main[l1] -= fdiff ./ sqrt(m.b_main_grad[l1])
            m.b_ctx[l2] -= fdiff ./ sqrt(m.b_ctx_grad[l2])


            # Gradients
            fdiff *= fdiff
            @inbounds for j = 1:vecsize
                m.W_main_grad[j, l1] += grad_main[j] ^ 2
                m.W_ctx_grad[j, l2] += grad_ctx[j] ^ 2
            end
            m.b_main_grad[l1] += fdiff
            m.b_ctx_grad[l2] += fdiff
        end

        if verbose && n % 10 == 0
            println("iteration ", n, " cost ", J)
        end
    end
end
