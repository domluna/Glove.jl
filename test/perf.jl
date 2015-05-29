import Glove

corpus = "data/corpus_test2.txt"
vocab = Glove.make_vocab(corpus)
comatrix = Glove.make_cooccur(vocab, corpus)
model = Glove.Model(comatrix, vecsize=100)
solver = Glove.Adagrad(500)

# JIT compile
@time Glove.fit!(model, solver)
@time Glove.fit!(model, solver)

