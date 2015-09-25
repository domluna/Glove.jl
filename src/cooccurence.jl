typealias CooccurenceDict{T<:AbstractFloat} DataStructures.DefaultDict{NTuple{2, Int}, T, zero(T)}

"""
make_cooccur creates the co-occurence matrix X. X is symmetric.
The `window_size` param determines how many of the surrounding tokens
should be considered a context token to the main token.
"""
function make_cooccur(table::LookupTable, filepath::AbstractString; window_size::Int=10)
    codict = DataStructures.DefaultDict(NTuple{2,Int}, Float64, 0.0)
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

                    p1 = tuple(mtok_id, ctok_id)
                    p2 = tuple(ctok_id, mtok_id)

                    # Symmetric context
                    codict[p1] += incr
                    codict[p2] += incr
                end
            end
        end
    end
    codict
end


function make_cooccur(table::LookupTable, tokens::Array; window_size::Int=10)
    codict = DataStructures.DefaultDict(NTuple{2,Int}, Float64, 0.0)
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

        p1 = tuple(mtok_id, ctok_id)
        # p2 = tuple(ctok_id, mtok_id)

        # Symmetric context
        codict[p1] += incr
        # codict[p2] += incr
      end
    end
    codict
end
