import GloVe

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

# 
function bench()
end

vocab = GloVe.make_vocab(corpus)
comatrix = GloVe.make_cooccur(vocab, corpus)
model = GloVe.Model(comatrix, vecsize=100)
solver = GloVe.Adagrad(1)

# 1 iteration
@time GloVe.train!(model, solver)
