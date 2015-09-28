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

"""
Each vocab word in associated with a word vector and a context vector.
The paper initializes the weights to values [-0.5, 0.5] / vecsize+1 and
the gradients to 1.0.

The +1 term is for the bias.
"""
function Model{T<:AbstractFloat}(::Type{T}, vocabsize::Int, vecsize::Int)
    const shift = T(0.5)
    Model(
        (rand(T, vecsize, vocabsize) - shift) / T(vecsize + 1),
        (rand(T, vecsize, vocabsize) - shift) / T(vecsize + 1),
        (rand(T, vocabsize) - shift) / T(vecsize + 1),
        (rand(T, vocabsize) - shift) / T(vecsize + 1),
        ones(T, vecsize, vocabsize),
        ones(T, vecsize, vocabsize),
        ones(T, vocabsize),
        ones(T, vocabsize),
    )
end

function adagrad!{T<:AbstractFloat}(
  ::Type{T},
  m::Model{T},
  covec::CooccurenceVector{T},
  epochs::Int;
  lrate=T(1e-2),
  xmax::Int=100,
  alpha=T(0.75))

    # store the cost per iteration
    costs = zeros(T, epochs)
    vecsize = size(m.W_main, 1)

    for n = 1:epochs
        for i = 1:length(covec)
            @inbounds pair = covec[i]
            midx = pair[1] # main index
            cidx = pair[2] # context index
            val = pair[3] # value

            diff = m.b_main[midx] + m.b_ctx[cidx] - log(val)
            @inbounds for j = 1:vecsize
              diff += m.W_main[j, midx] * m.W_ctx[j, cidx]
            end

            fdiff = ifelse(val < xmax, (val / xmax) ^ alpha, T(1)) * diff
            costs[n] = T(0.5) * fdiff * diff

            # Adaptive learning gradient updates
            fdiff *= lrate
            @inbounds for j = 1:vecsize
                tm = fdiff * m.W_ctx[j, cidx]
                tc = fdiff * m.W_main[j, midx]
                m.W_main[j, midx] -= tm / sqrt(m.W_main_grad[j, midx])
                m.W_ctx[j, cidx] -= tc / sqrt(m.W_ctx_grad[j, cidx])
                m.W_main_grad[j, midx] += tm * tm
                m.W_ctx_grad[j, cidx] += tc * tc
            end

            # bias updates
            m.b_main[midx] -= fdiff / sqrt(m.b_main_grad[midx])
            m.b_ctx[cidx] -= fdiff / sqrt(m.b_ctx_grad[cidx])
            fdiff *= fdiff
            m.b_main_grad[midx] += fdiff
            m.b_ctx_grad[cidx] += fdiff
        end
    end
    costs
end
