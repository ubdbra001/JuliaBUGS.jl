"""
Markov Melding for `BUGSModel`s, based on Goudie, Presanis, Lunn, De Angelis and Wernisch,
*Joining and Splitting Models with Markov Melding*, Bayesian Analysis 14(1), 2019

Several submodels that share a link parameter are 'melded' into a single target density with 
the following posterior distribution:

```math
p_{\\text{meld}}(\\phi, \\psi_1, \\ldots, \\psi_M \\mid y_1, \\ldots, y_M)
  \\;\\propto\\; p_{\\text{pool}}(\\phi) \\prod_{m=1}^{M} \\frac{p_m(\\phi, \\psi_m, y_m)}{p_m(\\phi)},
```

where ``p_{\\text{pool}}`` is a pooling of the submodel marginals.
"""
module Melding

# Pooling options

"""
    Abstract Pooling

A rule the combines the submodel prior marginals of the link parameter into the pooled prior
"""
abstract type AbstractPooling end

"""
    ProductOfExperts()

Product-of-experts pooling: ``p_{\\text{pool}}(\\phi) \\propto \\prod_m p_m(\\phi)``.
Every ``p_m(\\phi)`` cancels, so the melded density is the product of the submodel joints and
no marginal of the link is ever evaluated.
"""
struct ProductOfExperts <: AbstractPooling end

"""
    Dictatorial(k::Int)

Dictatorial pooling, ``p_{\\text{pool}}(\\phi) = p_k(\\phi)``: submodel `k` keeps its prior on
the link and every other submodel has its own prior on the link divided out.
"""
struct Dictatorial <: AbstractPooling
    dictator::Int
    function Dictatorial(k::Integer)
        k >= 1 || throw(ArgumentError("the dictator must be positive, got $k"))
        return new(Int(k))
    end
end


end