#
#

typealias Token Union(ASCIIString, UTF8String, SubString{ASCIIString}, SubString{UTF8String})

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

# Generates the co-occurence matrix X. X is symmetric.
function make_cooccur(v::Vocab, filename::String; window::Int=10)
    comatrix = spzeros(length(v), length(v))
    open(filename) do f
        for line in eachline(f)
            tokens = split(line)
            for i = 1:length(tokens)
                @inbounds t = tokens[i]
                c_id = v[t]
                lwindow = tokens[max(1, i-window):i-1]

                for li = 1:length(lwindow)
                    @inbounds lt = lwindow[li]
                    l_id = v[lt]

                    # words that are d spaces spart, contribute
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

