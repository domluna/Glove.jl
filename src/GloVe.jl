module GloVe

import Base: getindex, length

export Model, Vocab, Token, Cooccurence,
       Solver, Adagrad,
       make_vocab, make_cooccur,
       train!


if VERSION < v"0.4.0-dev"
    using Docile
end

# Not sure if I'll need this yet.
using Compat

include("corpus.jl")
include("model.jl")

end
