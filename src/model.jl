# Cooccurence represents a co-occurence between two tokens.
# i is the id of the main token, j is the id of the context token.
# v is the co-occurence value between the two tokens.
immutable Cooccurence{Ti,Tj<:Int, T}
    i::Ti
    j::Tj
    v::T
end

# make_covector creates an Vector of Cooccurence's.
#
# TODO: Once Sparse Matrices are good enough this
# will no longer be required.
function make_covector{T}(comatrix::SparseMatrixCSC{T})
    aa = findnz(comatrix)
    n = length(aa[1])
    a = Array(Cooccurence{Int, Int, T}, n)
    @inbounds for i = 1:n
        a[i] = Cooccurence(aa[1][i], aa[2][i], aa[3][i])
    end
    a
end

# Solver represents a solver using some variation of Gradient Descent.
# http://en.wikipedia.org/wiki/Gradient_descent
abstract Solver

# Adagrad is the method used for optimization in the paper. However,
# other methods may show better results.
#
# http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
type Adagrad{T} <: Solver
    epochs::Int
    lrate::T
end

# 0.05 is the learning rate used in the paper
Adagrad(epochs) = Adagrad(epochs, 0.05)
Adagrad(epochs, lrate) = Adagrad(epochs, lrate)

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

# fit! fits the Model to the data through the gradient descent variation.
function fit!(m::Model, s::Adagrad; xmax=100, alpha=0.75)
    J = 0.0

    shuffle!(m.covec)

    vecsize = size(m.W_main, 1)
    S = eltype(m.b_main)
    vm = zeros(S, vecsize)
    vc = zeros(S, vecsize)
    grad_main = zeros(S, vecsize)
    grad_ctx = zeros(S, vecsize)

    for n = 1:s.epochs
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
            fdiff = ifelse(v < xmax, (v / xmax) ^ alpha, one(T)) * diff
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

        println("Iteration ", n, ", cost = ", J)
    end
end

# similar_words returns the n most similar words
function similar_words{T, S<:Token}(M::Matrix{T}, v::Vocab, word::S; n::Int=10)
    c_id = v[word]

    dists = vec(M[:, c_id]' * M) / norm(M[:, c_id]) / norm(M, 1)

    sorted_ids = sortperm(dists, rev=true)[1:n+1]
    sim_words = Token[]

    for id = sorted_ids
        if c_id == id
            continue
        end
        word = v[id]
        push!(sim_words, word)
    end
    sim_words
end
