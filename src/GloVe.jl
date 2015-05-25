module GloVe

import Base: haskey, getindex, setindex! 

if VERSION < v"0.4.0-dev"
    using Docile
end

# Not sure if I'll need this yet.
using Compat

include("corpus.jl")
include("model.jl")
include("util.jl")

end
