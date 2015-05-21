# Utility functions to make interacting
# with the GloVe model a bit easier
# right off the bat.

# combine averages the main and context matrices/bias vectors. 
function combine(m::Model)
    vecsize = size(m.W_main, 2)
    M = m.W_main + m.W_ctx .+ (m.b_main / vecsize) .+ (m.b_ctx / vecsize)
    M /= (vecsize + 1)
    M
end

# similar_words returns the n most similar words.
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

