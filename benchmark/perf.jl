#
# Mimics the setup in demo.sh for Stanford's C implementation.
# Roughly ~3x slower than the C model as of 09/28/15
#
# The dataset is from char-rnn.
#
# To Download go to
using Glove
using DataStructures

S = Float64
println("Using type: ", S)

file = "input.txt"
tokens = split(read(file, String))
println("Counted ", length(tokens), " tokens")
ct = counter(tokens)
println("Counted ", length(ct.map), " unique tokens")
filter!((e) -> e.second >= 5, ct.map)
println("Counted ", length(ct.map), " filtered tokens")

filtered_tokens = collect(keys(ct))
table = LookupTable(filtered_tokens)

println("Creating Co-occurences")

println("before jit...")
@time codict = weightedsums(S, table, tokens, window=15)
println("after jit ...")
@time codict = weightedsums(S, table, tokens, window=15)

# make the covec
n = length(codict)
covec = CooccurenceVector{S}(undef, n)
cokeys = collect(keys(codict))
covals = collect(values(codict))
@inbounds for i=1:n
  covec[i] = (cokeys[i][1], cokeys[i][2], covals[i])
end

vocabsize = length(table)
vecsize = 50
model = Model(S, vocabsize, vecsize)

println("before jit ...")
@time adagrad!(S, model, covec, 1, xmax=10)
println("after jit ...")
@time adagrad!(S, model, covec, 1, xmax=10)
