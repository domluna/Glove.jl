# read_stanford reads a file where each line is a word
# and its corresponding word vector.
# 
# Each line starts off with the word followed by n floating
# point values where n is the word vector size.
#
# Ex:
#
# the 0.80384 -1.0366 ... -1.0806 0.84718 -0.36196 
#
# M is a Matrix of size (vector_size, vocab_size).
# 
# returns a Vocab
#
function read_stanford{T}(filename::String, M::Matrix{T})
    v = Vocab()
    open(filename) do f
        for line in eachline(f)
            wordvec = split(line)
            word = wordvec[1]

            id = v[word]
            @inbounds for i = 1:length(wordvec)-1
                M[i, id] = parse(T, wordvec[i+1])
            end
        end
    end
    v
end
