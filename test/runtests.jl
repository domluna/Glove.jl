reload("Glove")
using Base.Test
using Compat

# Test from python Glove implementation.
# https://github.com/maciejkula/glove-python/tree/master/glove/tests
corpus1 = "data/corpus_test1.txt"
vocab = Glove.make_vocab(corpus1)
@test vocab.word2id == @compat Dict("a" => 1, "naive" => 2, "fox" => 3)
@test vocab.id2word == @compat Dict(1 => "a", 2 => "naive",  3 => "fox")
@test length(vocab) == 3
@test vocab["fox"] == 3
@test vocab[1] == "a"

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
Glove.fit!(model, solver)

# paper recommends adding the main and context matrices.
M = model.W_main + model.W_ctx
top_words = Glove.similar_words(M, vocab, "trees", n=10)[1:3]
@test in("graph", top_words)
top_words = Glove.similar_words(M, vocab, "graph", n=10)[1:3]
@test in("trees", top_words)
