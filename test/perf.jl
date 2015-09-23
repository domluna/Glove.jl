import Glove


function bench_read_corpus()
end

function bench_create_corpus()
end

function bench_model()
end

corpus = "data/stanford_test.txt"
vocab = Glove.make_vocab(corpus)
comatrix = Glove.make_cooccur(vocab, corpus)
model = Glove.Model(comatrix, vecsize=100)
solver = Glove.Adagrad(500) # 500 iters

# JIT compile
@time Glove.fit!(model, solver)
# Real timing
@time Glove.fit!(model, solver)
