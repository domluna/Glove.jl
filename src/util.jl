# Utility functions to make interacting
# with the GloVe model a bit easier
# right off the bat.

# combine averages the main and context matrices/bias vectors. 
function combine(m::Model)
    vecsize = size(m.W_main, 1)
    vs = size(m.W_main, 2)
    M = similar(m.W_main)

    @inbounds for j = 1:vs, i = 1:vecsize
        M[i, j] = m.W_main[i, j] + m.W_ctx[i, j] #+ (m.b_main[j] + m.b_ctx[j]) / vecsize
    end
    M
end

# similarity_matrix computes the similarities between all words.
function similarity_matrix{T}(M::Matrix{T})
    SM = similar(M)
    vecsize = size(M, 1)
    vs = size(M, 2)

    for j = 1:vs, i = 1:vecsize
    end
    SM
end

# similar_words returns the n most similar words.
function similar_words{T}(M::Matrix{T}, vocab, id2word, word; n=10)
    c_id = vocab[word]

    dists = vec(M[:, c_id]' * M) / norm(M[:, c_id]) / norm(M, 1)
    sorted_ids = sortperm(dists, rev=true)[1:n+1]
    sim_words = String[]

    for id = sorted_ids
        if c_id == id
            continue
        end
        word = id2word[id]
        push!(sim_words, word)
    end
    sim_words
end

