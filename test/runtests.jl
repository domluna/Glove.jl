using Glove
using Test
using LinearAlgebra

# Anything <: AbstractFloat, just here for easy use
S = Float32

# Test from python Glove implementation.
# https://github.com/maciejkula/glove-python/tree/master/glove/tests
corpus1 = split(read("data/corpus_test1.txt", String))
table = LookupTable(corpus1)
@test table.word2id == Dict("a" => 1, "naive" => 2, "fox" => 3)
@test table.id2word == Dict(1 => "a", 2 => "naive",  3 => "fox")
@test length(table) == 3
@test table["fox"] == 3
@test table[1] == "a"

# create the co-occurence matrix and table in 1 pass on the corpus.
codict = weightedsums(S, table, corpus1)
@test codict[(1,1)] == 0.0
@test codict[(1,2)] == 1.0
@test codict[(1,3)] == 0.5

@test codict[(2,1)] == 1.0
@test codict[(2,2)] == 0.0
@test codict[(2,3)] == 1.0

@test codict[(3,1)] == 0.5
@test codict[(3,2)] == 1.0
@test codict[(3,3)] == 0.0

corpus2 = split(read("data/corpus_test2.txt", String))
table = LookupTable(corpus2)
codict = weightedsums(S, table, corpus2)

cokeys = collect(keys(codict))
covals = collect(values(codict))

n = length(codict)
covec = CooccurenceVector{S}(undef, n)

@inbounds for i=1:n
  covec[i] = (cokeys[i][1], cokeys[i][2], covals[i])
end

vocabsize = length(table)
vecsize = 10

model = Model(S, vocabsize, vecsize)

@test typeof(model) == Glove.Model{S}
@test eltype(model.W_main) == S
@test typeof(covec) == CooccurenceVector{S}
@test eltype(covec) == Tuple{Int, Int, S}


# The corpus is very, very small.
# So even with a large amount of iterations
# this test may fail.
adagrad!(S, model, covec, 500)

# make sure everything is still the same type
@test eltype(model.W_main) == S

# similar words
function similar_words(
  M::Matrix{T},
  table::LookupTable,
  word::S;
  n::Int=10) where {T, S<:Token}

    c_id = table[word]

    dists = vec(M[:, c_id]' * M) / norm(M[:, c_id]) / norm(M, 1)

    sorted_ids = sortperm(dists, rev=true)[1:n+1]
    sim_words = Token[]

    for id = sorted_ids
        if c_id == id
            continue
        end
        word = table[id]
        push!(sim_words, word)
    end
    sim_words
end

# paper recommends adding the main and context matrices.
M = model.W_main + model.W_ctx
top_words = similar_words(M, table, "trees", n=10)[1:3]
@test in("graph", top_words)
top_words = similar_words(M, table, "graph", n=10)[1:3]
@test in("trees", top_words)
