using Glove
using Base.Test

# Test from python Glove implementation.
# https://github.com/maciejkula/glove-python/tree/master/glove/tests
corpus1 = split(readall("data/corpus_test1.txt"))
table = LookupTable(corpus1)
@test table.word2id == Dict("a" => 1, "naive" => 2, "fox" => 3)
@test table.id2word == Dict(1 => "a", 2 => "naive",  3 => "fox")
@test length(table) == 3
@test table["fox"] == 3
@test table[1] == "a"

# create the co-occurence matrix and table in 1 pass on the corpus.
# codict = getcounts(table, corpus1)
codict = getcounts(Float64, table, corpus1)
# @test full(codict) == [0.0 1.0 0.5; 1.0 0.0 1.0; 0.5 1.0 0.0]
@test codict[(1,1)] == 0.0
@test codict[(1,2)] == 1.0
@test codict[(1,3)] == 0.5

@test codict[(2,1)] == 1.0
@test codict[(2,2)] == 0.0
@test codict[(2,3)] == 1.0

@test codict[(3,1)] == 0.5
@test codict[(3,2)] == 1.0
@test codict[(3,3)] == 0.0

# corpus2 = split(readall("data/corpus_test2.txt"))
# table = LookupTable(corpus2)
# codict = getcounts(table, corpus2)
#
# model = Glove.Model(codict, vecsize=10)
#
# cokeys = collect(keys(codict))
# covals = collect(values(codict))
#
# n = length(codict)
# covec = CooccurenceVector{Float64}(n)
#
# for i=1:n
#   covec[i] = (cokeys[i], covals[i])
# end
#
# # The corpus is very, very small.
# # So even with a large amount of iterations
# # this test may fail.
# adagrad!(model, epochs=1000)
#
# # paper recommends adding the main and context matrices.
# M = model.W_main + model.W_ctx
# top_words = Glove.similar_words(M, table, "trees", n=10)[1:3]
# @test in("graph", top_words)
# top_words = Glove.similar_words(M, table, "graph", n=10)[1:3]
# @test in("trees", top_words)
