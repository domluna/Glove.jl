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
    W::Matrix{T}
    Wgrad::Matrix{T}
    b::Vector{T}
    bgrad::Vector{T}
    covec::Vector{Cooccurence}
    vs::Int
end

# Each vocab word in associated with a word vector and a context vector.
# word vectors are in rows 1..vs and context vectors are rows 1+vs..2vs
function Model(comatrix; vecsize=100)
    vs = size(comatrix)[1]
    Model(
        (rand(vs*2, vecsize) - 0.5) / vecsize,
        rand(vs*2, vecsize),
        (rand(vs*2) - 0.5) / vecsize,
        rand(vs*2),
        CoVector(comatrix),
        vs
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
            l1 = co.i
            l2 = m.vs + co.j

            diff = m.W[l1,:] * m.W[l2,:]' + m.b[l1] + m.b[l2] - log(co.v)
            fdiff = ifelse(co.v < xmax, (co.v / xmax) ^ alpha, 1.0) * diff
            J += 0.5 * fdiff * diff

            grad_main = fdiff * m.W[l2, :]
            grad_ctx = fdiff * m.W[l1, :] 

            # Adaptive learning
            m.W[l1, :] -= s.lrate * grad_main ./ sqrt(m.Wgrad[l1, :])
            m.W[l2, :] -= s.lrate * grad_ctx ./ sqrt(m.Wgrad[l2, :])
            m.b[l1, :] -= s.lrate * fdiff ./ sqrt(m.bgrad[l1, :])
            m.b[l2, :] -= s.lrate * fdiff ./ sqrt(m.bgrad[l2, :])

            m.Wgrad[l1, :] += grad_main .^ 2
            m.Wgrad[l2, :] += grad_ctx .^ 2
            fdiff *= fdiff
            m.bgrad[l1, :] += fdiff
            m.bgrad[l2, :] += fdiff

        end

        if n % 10 == 0
            println("iteration $n, cost $J")
        end
    end
    # Average the main and context vectors
    M = m.W[1:m.vs, :] + m.W[m.vs+1:end, :]
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

