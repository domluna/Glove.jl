module GloVe

if VERSION < v"0.4.0-dev"
    using Docile
end

using Compat

include("corpus.jl")
include("model.jl")
include("util.jl")

end
