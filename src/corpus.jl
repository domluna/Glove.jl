#
#

typealias Token Union(ASCIIString, UTF8String, SubString{ASCIIString}, SubString{UTF8String})

# Vocab is a self-counting dictionary.
# A Token is added to the Vocab through
# the getindex operation.
# 
# Ex:
#   v = Vocab()
#   id = v["Batman"]
#
# If the token is new to the vocabulary then the id counter is incremented.
# Otherwise the original id is returned and no changes are made.
type Vocab 
    d::Dict{Token, Int}
    counter::Int
end
Vocab() = Vocab(Dict{Token, Int}(), 1)
function getindex(v::Vocab, t::Token)  
    id = get!(v.d, t, v.counter)
    if id == v.counter # increment counter
        v.counter += 1
    end
    id
end
length(v::Vocab) = length(v.d)

# make_id2word creates a Dict. Flipping the keys/values
# of the Vocab.
# keys -> values && values -> keys
# 
# This is useful when comparing the similarity of words.
# In this situation a Vocab does not suffice.
function make_id2word(v::Vocab)
    id2word = Dict{Int, Token}()
    for (w, id) = v.d
        id2word[id] = w
    end
    id2word
end

# make_vocab creates the vocabulary from the corpus.
# Each unique token in the corpus is assigned a unique id.
function make_vocab(filename::String)
    v = Vocab()
    open(filename) do f
        for line in eachline(f)
            tokens = line |> strip |> split
            for t = tokens
                v[t]
            end
        end
    end
    v
end

# make_cooccur creates the co-occurence matrix X. X is symmetric.
# The window_size determines how many of the surrounding tokens should
# be guaranteed as a context token to the main token.
function make_cooccur(v::Vocab, filename::String; window_size::Int=10)
    comatrix = spzeros(length(v), length(v))
    open(filename) do f
        for line in eachline(f)
            tokens = line |> strip |> split
            for i = 1:length(tokens)
                @inbounds t = tokens[i]
                c_id = v[t]
                lwindow = tokens[max(1, i-window_size):i-1]

                for li = 1:length(lwindow)
                    @inbounds lt = lwindow[li]
                    l_id = v[lt]

                    # tokens that are d spaces spart, contribute
                    # 1.0 / d to the total count.
                    incr = 1.0 / (i - li)

                    # symmetry
                    @inbounds comatrix[c_id, l_id] += incr
                    @inbounds comatrix[l_id, c_id] += incr
                end
            end
        end
    end
    comatrix
end

