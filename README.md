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

### Caveats

1. File I/O is very slow

Temporary solution. Load the file into main memory and go from there.

```julia
using Glove

v = Vocab()
corpus = split(readall("file"))

@inbounds for i = 1:length(corpus)
    v[corpus[i]]
end
```
2. Sparse matrices are very slow

For now I think using a Dict would be best.

Hopefully these will get faster soon but for now I would recommend
not to use it on large datasets.

### TODO

1. Benchmark vs C implementation
2. Nice notebook example
