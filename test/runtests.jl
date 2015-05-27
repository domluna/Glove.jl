import Glove
using Base.Test
using Compat

# Test from python Glove implementation.
# https://github.com/maciejkula/glove-python/tree/master/glove/tests
corpus1 = "data/corpus_test1.txt"
vocab = Glove.make_vocab(corpus1)
@test vocab.counter == 4
@test vocab.d == @compat Dict("a" => 1, "naive" => 2, "fox" => 3)

# create the co-occurence matrix and vocab in 1 pass on the corpus.
comatrix = Glove.make_cooccur(vocab, corpus1)
@test full(comatrix) == [0.0 1.0 0.5; 1.0 0.0 1.0; 0.5 1.0 0.0]

corpus2 = "data/corpus_test2.txt"
vocab = Glove.make_vocab(corpus2)
comatrix = Glove.make_cooccur(vocab, corpus2)
model = Glove.Model(comatrix, vecsize=10)

# The corpus is very, very small.
# So even with a large amount of iterations
# this test may fail.
solver = Glove.Adagrad(1000)
Glove.train!(model, solver)

id2word = Dict{Int, Glove.Token}()
for (w, id) = vocab.d
    id2word[id] = w
end

# similar_words returns the n most similar words
function similar_words{T}(M::Matrix{T}, v::Glove.Vocab, id2word, word; n=10)
    c_id = v[word]

    dists = vec(M[:, c_id]' * M) / norm(M[:, c_id]) / norm(M, 1)
    
    sorted_ids = sortperm(dists, rev=true)[1:n+1]
    sim_words = Glove.Token[]

    for id = sorted_ids
        if c_id == id
            continue
        end
        word = id2word[id]
        push!(sim_words, word)
    end
    sim_words
end

# model is trained
M = model.W_main + model.W_ctx
top_words = similar_words(M, vocab, id2word, "trees", n=10)[1:3]
@test in("graph", top_words)
top_words = similar_words(M, vocab, id2word, "graph", n=10)[1:3]
@test in("trees", top_words)
