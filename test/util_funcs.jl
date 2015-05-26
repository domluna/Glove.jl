# similar_words returns the n most similar words.

# read_stanford reads a file where each line is a word
# and its corresponding vector.
# 
# Each line starts off with the word followed by n floating
# point values where n is the word vector size.

# m is a Matrix of size (vector_size, vocab_size)
# 
# Ex:
#
# the 0.80384 -1.0366 ... -1.0806 0.84718 -0.36196 
function read_stanford{T}(filename::String, m::Matrix{T}; binary::Bool=false)
    id = 1
    vocab = Dict{Union(ASCIIString, UTF8String), Int}()
    open(filename) do f
        for line in eachline(f)
            line = split(line)
            word = line[1]

            vocab[word] = id
            @inbounds for i = 1:length(line)-1
                M[i, id] = parse(T, line[i+1])
            end

            id += 1
        end
    end
    vocab
end
