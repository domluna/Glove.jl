import GloVe
using Base.Test

# Test from python GloVe implementation.
# https://github.com/maciejkula/glove-python/tree/master/glove/tests
vocab = Dict(["a", "naive", "fox"], [1,2,3])
corpus = [["a naive fox"]]

comatrix = GloVe.make_cooccur(vocab, corpus)
expected = [0.0 1.0 0.5; 1.0 0.0 1.0; 0.5 1.0 0.0]

@test full(comatrix) == expected

# Mock corpus (from Gensim word2vec tests)
corpus = split("""human interface computer
survey user computer system response time
eps user interface system
system human system eps
user response time
trees
graph trees
graph minors trees
graph minors survey
I like graph and stuff
I like trees and stuff
Sometimes I build a graph
Sometimes I build trees""", '\n')

vocab = GloVe.make_vocab(corpus)
comatrix = GloVe.make_cooccur(vocab, corpus)
model = GloVe.Model(comatrix, vecsize=10)

# run for 500 iterations
solver = GloVe.Adagrad(500)
GloVe.train!(model, solver)

id2word = Dict()
for (w, id) = vocab
    id2word[id] = w
end

# model is trained
M = GloVe.combine(model)
top_word = GloVe.similar_words(M, vocab, id2word, "trees", n=1)[1]
@test top_word == "graph"
top_word = GloVe.similar_words(M, vocab, id2word, "graph", n=1)[1]
@test top_word == "trees"

