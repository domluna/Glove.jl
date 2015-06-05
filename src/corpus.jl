#

typealias Token Union(ASCIIString, UTF8String, SubString{ASCIIString}, SubString{UTF8String})

# Vocab is a self-counting dictionary.
# It maps words to ids and ids to words.
# 
type Vocab
    id2word::Dict{Int, Token}
    word2id::Dict{Token, Int}
end

Vocab() = Vocab(Dict{Int, Token}(), Dict{Token, Int}())

# Inserts the word in the Vocab.
# If the word is already in the Vocab
# nothing is done.
function insert!{T<:Token}(v::Vocab, word::T)
    if haskey(v.word2id, word)
        # Not sure if this useful
        #= error("$word already in Vocab.") =#
        return
    end

    id = length(v.id2word) + 1
    v.word2id[word] = id
    v.id2word[id] = word
end

getindex(v::Vocab, id::Int) = getindex(v.id2word, id)
getindex{T<:Token}(v::Vocab, word::T) = getindex(v.word2id, word)
length(v::Vocab) = length(v.id2word)

# Creates the Vocab from a textfile.
function make_vocab(filename::String)
    v = Vocab()
    open(filename) do f
        for line = eachline(f)
            tokens = split(line)
            @inbounds for i = 1:length(tokens)
                insert!(v, tokens[i])
            end
        end
    end
    v
end

# Creates the Vocab from a String array in memory.
function make_vocab{T<:Token}(corpus::Vector{T})
    v = Vocab()
    @inbounds for i = 1:length(corpus)
        insert!(v, tokens[i])
    end
    v
end

# make_cooccur creates the co-occurence matrix X. X is symmetric.
# The window_size determines how many of the surrounding tokens should
# be guaranteed as a context token to the main token.
function make_cooccur(v::Vocab, filename::String; window_size::Int=10)
    comatrix = spzeros(length(v), length(v))
    open(filename) do f
        for line = eachline(f)
            tokens = split(line)
            for i = 1:length(tokens)
                @inbounds t = tokens[i]
                c_id = v[t]

                for li = max(1, i - window_size):i-1
                    @inbounds lt = tokens[li]
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


function make_cooccur{T<:Token}(v::Vocab, corpus::Vector{T}; window_size::Int=10)
    comatrix = spzeros(length(v), length(v))
    for i = 1:length(corpus)
        @inbounds token = corpus[i]
        c_id = v[token]

        for li = max(1, i - window_size):i-1
            @inbounds lt = corpus[li]
            l_id = v[lt]

            # tokens that are d spaces spart, contribute
            # 1.0 / d to the total count.
            incr = 1.0 / (i - li)

            # symmetry
            @inbounds comatrix[c_id, l_id] += incr
            @inbounds comatrix[l_id, c_id] += incr
        end
        i % 10_000 == 0 && println("Done iteration ", i)
    end
    comatrix
end

