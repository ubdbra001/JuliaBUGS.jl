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


end