isdefined(Base, :__precompile__) && __precompile__(true)

module Glove

import Base: insert!
import DataStructures

export Model, LookupTable, Token,
       Solver, Adagrad,
       make_vocab, make_cooccur,
       similar_words, fit!


if VERSION < v"0.4.0-dev"
    using Docile
end

# Not sure if I'll need this yet.
using Compat

include("lookuptable.jl")
include("cooccurence.jl")
include("model.jl")

end
