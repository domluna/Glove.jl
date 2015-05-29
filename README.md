Glove
-----

[![Build Status](https://travis-ci.org/domluna/GloVe.jl.svg?branch=master)](https://travis-ci.org/domluna/GloVe.jl)

Implements [Global Word Vectors](http://nlp.stanford.edu/projects/glove/).

### Usage

```julia
using Glove

# Path to some text file.
corpus = "somefile"

vocab = Glove.make_vocab(corpus)
comatrix = Glove.make_cooccur(vocab, corpus)

# Word vectors will be represented by 25 floating point values.
model = Glove.Model(comatrix, vecsize=25)

# Run Adagrad for 50 epochs.
solver = Glove.Adagrad(50)

fit!(model, solver, verbose=true)

# We now have a fit Glove model!
```

### TODO

1. Benchmark vs C implementation
2. Nice notebook
