#
# Co-occurence data structures
#
CooccurenceDict{T<:AbstractFloat} = DataStructures.DefaultDict{NTuple{2, Int}, T, T}
CooccurenceVector{T<:AbstractFloat} = Vector{Tuple{Int, Int, T}}

# TODO: better name
# TODO: not sure how the coccurences thing works here yet
"""
weightedsums counts the co-occurences as specified
in the GloVe paper.

Returns a `CooccurenceDict` with the weighted co-occurence sum.
"""
function weightedsums(::Type{T}, table::LookupTable, tokens::Vector;
  window::Int=10) where T<:AbstractFloat

    codict = CooccurenceDict{T}(T(0.0))

    for i = 1:length(tokens)
      @inbounds mtok = tokens[i]
      # skip tokens not in the table
      if !haskey(table, mtok)
        continue
      end
      mtok_id = table[mtok]

      for j = max(1, i - window):i-1
        @inbounds ctok = tokens[j]
        if !haskey(table, ctok)
          continue
        end
        ctok_id = table[ctok]

        # The farther away the context token is from the
        # main token the less it contributes
        dist = i - j
        incr = T(1.0 / dist)

        p1 = tuple(mtok_id, ctok_id)
        p2 = tuple(ctok_id, mtok_id)

        # Symmetric context
        codict[p1] += incr
        codict[p2] += incr
      end
    end
    codict
end

# Reading from files is kinda of slow still. Recommending to just use readall
# TODO: make this faster
# function make_cooccur(table::LookupTable, filepath::AbstractString; window::Int=10)
#     codict = DataStructures.DefaultDict(NTuple{2,Int}, Float64, 0.0)
#     open(filepath) do f
#         for line = eachline(f)
#             tokens = split(line)
#             for i = 1:length(tokens)
#                 @inbounds mtok = tokens[i]
#                 mtok_id = table[mtok]
#
#                 for j = max(1, i - window):i-1
#                     @inbounds ctok = tokens[j]
#                     ctok_id = table[ctok]
#
#                     # The farther away the context token is from the
#                     # main token the less it contributes
#                     dist = i - j
#                     incr = 1.0 / dist
#
#                     p1 = tuple(mtok_id, ctok_id)
#                     p2 = tuple(ctok_id, mtok_id)
#
#                     # Symmetric context
#                     codict[p1] += incr
#                     codict[p2] += incr
#                 end
#             end
#         end
#     end
#     codict
# end
