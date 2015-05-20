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
function Model(comatrix; vecsize=100)
    vs = size(comatrix, 1)
    Model(
        (rand(vs, vecsize) - 0.5) / vecsize,
        (rand(vs, vecsize) - 0.5) / vecsize,
        (rand(vs) - 0.5),
        (rand(vs) - 0.5),
        (rand(vs, vecsize) - 0.5) / vecsize,
        (rand(vs, vecsize) - 0.5) / vecsize,
        (rand(vs) - 0.5),
        (rand(vs) - 0.5),
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

            grad_main = fdiff * m.W_ctx[l2, :]
            grad_ctx = fdiff * m.W_main[l1, :] 

            # Adaptive learning
            m.W_main[l1, :] -= s.lrate * grad_main ./ sqrt(m.W_main_grad[l1, :])
            m.W_ctx[l2, :] -= s.lrate * grad_ctx ./ sqrt(m.W_ctx_grad[l2, :])

            m.b_main[l1, :] -= s.lrate * fdiff ./ sqrt(m.b_main_grad[l1, :])
            m.b_ctx[l2, :] -= s.lrate * fdiff ./ sqrt(m.b_ctx_grad[l2, :])

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

    # Average the main and context vectors
    M = m.W_main[1:end, :] + m.W_ctx[1:end, :]
    M /= size(M, 2)
    M
end

function similar_words(M::Matrix, vocab, id2word, word; n=10)
    c_id = vocab[word]
    dists = vec(M * M[c_id, :]') / norm(M[c_id, :]) / norm(M, 2)
    sorted_ids = sortperm(dists, rev=true)[1:n+1]
    sim_words = Any[]

    for i=1:length(sorted_ids)
        id = sorted_ids[i]
        if c_id == id
            continue
        end
        word = id2word[id]
        push!(sim_words, word)
    end
    sim_words
end

