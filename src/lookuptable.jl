import Base.convert

Token = Union{String, SubString{String}}

"""
LookupTable is a self-counting dictionary.
It maps words to ids and ids to words.
"""
struct LookupTable
    id2word::Dict{Int, Token}
    word2id::Dict{Token, Int}

    LookupTable() = new(Dict{Int, Token}(), Dict{Token, Int}())

    function LookupTable(tokens::Vector)
        table = LookupTable()
        @inbounds for i = 1:length(tokens)
          insert!(table, tokens[i])
        end
        return table
    end
end

function insert!(table::LookupTable, word::T) where T<:Token
    if !haskey(table.word2id, word)
      id = length(table.id2word) + 1
      table.word2id[word] = id
      table.id2word[id] = word
    end
end

getindex(table::LookupTable, word::T) where T<:Token = getindex(table.word2id, word)
getindex(table::LookupTable, id::Int) = getindex(table.id2word, id)
haskey(table::LookupTable, word::T) where T<:Token = haskey(table.word2id, word)
haskey(table::LookupTable, id::Int) = haskey(table.id2word, id)
length(table::LookupTable) = length(table.id2word)
