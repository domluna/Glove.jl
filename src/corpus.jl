
# Build a vocab from a corpus, vocab is a Dictionary
# mapping words to ids.
function make_vocab{T<:String}(corpus::Vector{T})
    vocab = Dict{T, Integer}()
    id = 1

    @inbounds for i=1:length(corpus) 
        line = split(strip(corpus[i]))
        for j=1:length(line)
            word = line[j]
            if !haskey(vocab, word)
                vocab[word] = id
                id += 1
            end
        end
    end
    vocab
end

# Generates the co-occurence matrix X. X is symmetric by default.
function make_cooccur(vocab, corpus; window=10)
    comatrix = spzeros(length(vocab), length(vocab))
    # fill the co-occurence matrix
    @inbounds for i=1:length(corpus)
        line = split(strip(corpus[i]))
        for j=1:length(line)
            lwindow = line[max(1, j-window):j-1]
            c_id = vocab[line[j]]

            for (li, lword) = enumerate(lwindow)
                l_id = vocab[lword]

                # words that are d spaces spart, contribute
                # 1.0 / d to the total count.
                incr = 1.0 / (j - li)

                # symmetry
                comatrix[c_id, l_id] += incr
                comatrix[l_id, c_id] += incr
            end
        end
    end
    comatrix
end

