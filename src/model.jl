# Glove model
type Model{T<:AbstractFloat}
    W_main::Matrix{T}
    W_ctx::Matrix{T}
    b_main::Vector{T}
    b_ctx::Vector{T}
    W_main_grad::Matrix{T}
    W_ctx_grad::Matrix{T}
    b_main_grad::Vector{T}
    b_ctx_grad::Vector{T}
end

# Each vocab word in associated with a word vector and a context vector.
# The paper initializes the weights to values [-0.5, 0.5] / vecsize+1 and
# the gradients to 1.0.
#
# The +1 term is for the bias.
function Model(vocabsize::Int, vecsize::Int)
    Model(
        (rand(vecsize, vocabsize) - 0.5) / (vecsize + 1),
        (rand(vecsize, vocabsize) - 0.5) / (vecsize + 1),
        (rand(vocabsize) - 0.5) / (vecsize + 1),
        (rand(vocabsize) - 0.5) / (vecsize + 1),
        ones(vecsize, vocabsize),
        ones(vecsize, vocabsize),
        ones(vocabsize),
        ones(vocabsize),
    )
end

function adagrad!(m::Model,
  covec::CooccurenceVector,
  epochs::Int;
  lrate=1e-2,
  xmax::Int=100,
  alpha=0.75)

    J = 0.0

    vecsize = size(m.W_main, 1)
    S = eltype(m.b_main)
    vm = zeros(S, vecsize)
    vc = zeros(S, vecsize)
    grad_main = zeros(S, vecsize)
    grad_ctx = zeros(S, vecsize)

    for n = 1:epochs
        # shuffle indices
        for i = 1:length(covec)
            @inbounds l1 = covec[i].i # main index
            @inbounds l2 = covec[i].j # context index
            @inbounds v = covec[i].v

            @inbounds for j = 1:vecsize
                vm[j] = m.W_main[j, l1]
                vc[j] = m.W_ctx[j, l2]
            end

            diff = dot(vec(vm), vec(vc)) + m.b_main[l1] + m.b_ctx[l2] - log(v)
            fdiff = ifelse(v < xmax, (v / xmax) ^ alpha, one(T)) * diff
            J += 0.5 * fdiff * diff

            fdiff *= lrate
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

# TODO: move this out into an example of how to use
# the final results.
function similar_words{T, S<:Token}(
  M::Matrix{T},
  table::LookupTable,
  word::S;
  n::Int=10)

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
