#

type LookupTable
    id2word::Dict{Int, String}
    word2id::Dict{String, Int}
end
LookupTable() = LookupTable(Dict{Int, String}(), Dict{String, Int}())
function LookupTable{T<:String}(words::Vector{T}, ids::Vector{Int})
    @assert length(words) == length(ids)
    LookupTable(Dict(zip(ids, words)), Dict(zip(words, ids)))
end
LookupTable{T<:String}(ids::Vector{Int}, words::Vector{T}) = LookupTable(words, ids)

==(l1::LookupTable, l2::LookupTable) = l1.id2word == l2.id2word && l1.word2id == l2.word2id
getindex(l::LookupTable, id::Int) = l.id2word[id]
getindex(l::LookupTable, word::String) = l.word2id[word]
haskey(l::LookupTable, id::Int) = haskey(l.id2word, id)
haskey(l::LookupTable, word::String) = haskey(l.word2id, word)

function setindex!(l::LookupTable, id::Int, word::String)
    l.id2word[id] = word
    l.word2id[word] = id
end
setindex!(l::LookupTable, word::String, id::Int) = setindex!(l, id, word)


# Build a vocab from a corpus, vocab is a Dictionary
# mapping words to ids.
function make_vocab{T<:String}(corpus::Vector{T})
    vocab = Dict{T, Integer}()
    id = 1

    @inbounds for i=1:length(corpus) 
        line = split(strip(corpus[i]))
        for j=1:length(line)
            word = line[j]
            if !haskey(vocab, word)
                vocab[word] = id
                id += 1
            end
        end
    end
    vocab
end

# Generates the co-occurence matrix X. X is symmetric by default.
function make_cooccur(vocab, corpus; window=10)
    comatrix = spzeros(length(vocab), length(vocab))
    # fill the co-occurence matrix
    @inbounds for i=1:length(corpus)
        line = split(strip(corpus[i]))
        for j=1:length(line)
            lwindow = line[max(1, j-window):j-1]
            c_id = vocab[line[j]]

            for (li, lword) = enumerate(lwindow)
                l_id = vocab[lword]

                # words that are d spaces spart, contribute
                # 1.0 / d to the total count.
                incr = 1.0 / (j - li)

                # symmetry
                comatrix[c_id, l_id] += incr
                comatrix[l_id, c_id] += incr
            end
        end
    end
    comatrix
end

