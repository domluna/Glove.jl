#
# Co-occurence data structures
#
typealias CooccurenceDict{T<:AbstractFloat} DataStructures.DefaultDict{NTuple{2, Int}, T, T}
# CooccurenceDict{T}(::Type{T}, default::T) = CooccurenceDict{T}(default)
typealias CooccurenceVector{T<:AbstractFloat} Vector{Tuple{Int, Int, T}}
# CooccurenceVector{T}(::Type{T}) = CooccurenceVector{T}()


"""
getcounts counts the co-occurences as specified
in the GloVe paper.

Returns a `CooccurenceDict` with the counts.
"""
function getcounts{T<:AbstractFloat}(::Type{T}, table::LookupTable,
  tokens::Vector;
  context_window::Int=10)

    codict = CooccurenceDict{T}(T(0.0))

    for i = 1:length(tokens)
      @inbounds mtok = tokens[i]
      mtok_id = table[mtok]

      for j = max(1, i - context_window):i-1
        @inbounds ctok = tokens[j]
        ctok_id = table[ctok]

        # The farther away the context token is from the
        # main token the less it contributes
        dist = i - j
        incr = T(1.0) / dist

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
# function make_cooccur(table::LookupTable, filepath::AbstractString; context_window::Int=10)
#     codict = DataStructures.DefaultDict(NTuple{2,Int}, Float64, 0.0)
#     open(filepath) do f
#         for line = eachline(f)
#             tokens = split(line)
#             for i = 1:length(tokens)
#                 @inbounds mtok = tokens[i]
#                 mtok_id = table[mtok]
#
#                 for j = max(1, i - context_window):i-1
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
