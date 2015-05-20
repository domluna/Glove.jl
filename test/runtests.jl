import GloVe
using Base.Test

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
id2word = Dict()
for (w, id) = vocab
    id2word[id] = w
end

cv = GloVe.make_cooccur(vocab, corpus)
M = GloVe.train!(GloVe.Model(cv, vecsize=10), GloVe.Adagrad(500))
sim_words = GloVe.similar_words(M, vocab, id2word, "trees", n=10)
println(sim_words)
@test "graph" in sim_words[1:2]

