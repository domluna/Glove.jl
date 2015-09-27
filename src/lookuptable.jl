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
function call(::Type{LookupTable}, tokens::Vector)
    table = LookupTable()
    @inbounds for i = 1:length(tokens)
      insert!(table, tokens[i])
    end
    table
end

"""
"""
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

# function Base.show(io::IO, lt::LookupTable)
#   println(io, "LookupTable with: ", length(lt), "elements")
#   println(io, "id2word lookup:")
#   println(io, lt.id2word)
#   println(io, "word2id lookup:")
#   println(io, lt.word2id)
# end
