import GloVe

corpus = "data/corpus_test2.txt"
vocab = GloVe.make_vocab(corpus)
comatrix = GloVe.make_cooccur(vocab, corpus)
model = GloVe.Model(comatrix, vecsize=100)
solver = GloVe.Adagrad(500)

# JIT compile
@time GloVe.train!(model, solver)
@time GloVe.train!(model, solver)

