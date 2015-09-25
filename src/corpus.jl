
# Redesign
"""
1. Get the user to work with LookupTable directly. Essentially it's just a utility
struct.

2. We should have a function that only reads a file once and builds the
vocab and co-occurence matrix or dict if it turns out sparse matrices are
too slow.

3. Aside from the above function we shouldn't much else; don't want to be too
opinionated. Provide the core data structures and let the user go from there.
"""

typealias Token Union(ASCIIString, UTF8String, SubString{ASCIIString}, SubString{UTF8String})

"""
LookupTable is a self-counting dictionary.
It maps words to ids and ids to words.
"""
type LookupTable
    id2word::Dict{Int, Token}
    word2id::Dict{Token, Int}
end

LookupTable() = LookupTable(Dict{Int, Token}(), Dict{Token, Int}())
function LookupTable(tokens::Array)
    table = LookupTable()
    @inbounds for i = 1:length(tokens)
      insert!(table, tokens[i])
    end
    table
end

function insert!{T<:Token}(table::LookupTable, word::T)
    if !haskey(table.word2id, word)
      id = length(table.id2word) + 1
      table.word2id[word] = id
      table.id2word[id] = word
    end
end
Base.getindex(table::LookupTable, id::Int) = getindex(table.id2word, id)
Base.getindex{T<:Token}(table::LookupTable, word::T) = getindex(table.word2id, word)
Base.length(table::LookupTable) = length(table.id2word)

"""
make_cooccur creates the co-occurence matrix X. X is symmetric.
The `window_size` param determines how many of the surrounding tokens
should be considered a context token to the main token.
"""
function make_cooccur(table::LookupTable, filepath::String; window_size::Int=10)
    comatrix = DataStructures.DefaultDict(NTuple{2,Int}, Float64, 0.0)
    open(filepath) do f
        for line = eachline(f)
            tokens = split(line)
            for i = 1:length(tokens)
                @inbounds mtok = tokens[i]
                mtok_id = table[mtok]

                for j = max(1, i - window_size):i-1
                    @inbounds ctok = tokens[j]
                    ctok_id = table[ctok]

                    # The farther away the context token is from the
                    # main token the less it contributes
                    dist = i - j
                    incr = 1.0 / dist

                    # Symmetric context
                    comatrix[mtok_id, ctok_id] += incr
                    comatrix[ctok_id, mtok_id] += incr
                end
            end
        end
    end
    comatrix
end

function make_cooccur(table::LookupTable, tokens::Array; window_size::Int=10)
    comatrix = DataStructures.DefaultDict(NTuple{2,Int}, Float64, 0.0)
    for i = 1:length(tokens)
      @inbounds mtok = tokens[i]
      mtok_id = table[mtok]

      for j = max(1, i - window_size):i-1
        @inbounds ctok = tokens[j]
        ctok_id = table[ctok]

        # The farther away the context token is from the
        # main token the less it contributes
        dist = i - j
        incr = 1.0 / dist

        # Symmetric context
        comatrix[mtok_id, ctok_id] += incr
        comatrix[ctok_id, mtok_id] += incr
      end
    end
    comatrix
end
