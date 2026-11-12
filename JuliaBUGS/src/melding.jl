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
using JuliaBUGS: model_parameters

using Distributions
using AbstractPPL: VarName

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

"""
    MeldedModel

The Markov melded target built by [`meld`](@ref).

# Fields
- `submodels`: The submodels passed to `meld`.
- `links`: The name of the link parameter in each submodel.
- `priors`: The priors for the link parameter in each submodel.
- `pooling`: The Pooling method used to meld the submodels.
- `param_indices`: Indices mapping the global sample to specific submodel parameters.
- `dimension`: Dimensions of the global sample.
"""
struct MeldedModel{P<:AbstractPooling}
    submodels::Vector
    links::Vector
    priors::Vector
    pooling::P
    param_indices::Vector
    dimension::Int
end

"""
    extract_prior(model, link_symbol::Symbol)

Extracts the prior distribution of the specified parameter from the model.
"""
function extract_prior(model, link_symbol::Symbol)
    vn = VarName{link_symbol}()
    node_info = model.g[vn]

    !node_info.is_stochastic && error("Link parameter $link_symbol must be a stochastic variable (~).")

    env = hasproperty(model, :evaluation_env) ? model.evaluation_env : NamedTuple()
    loop_vars = hasproperty(node_info, :loop_vars) ? node_info.loop_vars : NamedTuple()

    return node_info.node_function(env, loop_vars)
end


"""
    build_param_map(submodels, links)

Walk over submodels to identify positions of parameters for extraction from sample.
"""
function build_param_map(submodels, links)
    link_vns = [VarName{l}() for l in links]

    curr_slot = 1

    model_vec = Vector{Vector{Int}}()

    for (model, link) in zip(submodels, link_vns)

        param_vec = Int[]

        for var in model_parameters(model)
            slot = (var == link) ? 1 : (curr_slot += 1)
            push!(param_vec, slot)
        end

        push!(model_vec, param_vec)
    end

    return Tuple(model_vec), curr_slot
end

"""
    extract_submodel_params(param_indices, theta_global::AbstractVector)

Extracts the parameter values for each submodel from the theta_global vector
"""
function extract_submodel_params(param_indices, theta_global::AbstractVector)
    return map(idx -> theta_global[idx], param_indices)
end

"""
    extract_submodel_params(melded::MeldedModel, theta_global::AbstractVector)

Convenience function for extracting submodel parameters
"""
extract_submodel_params(melded::MeldedModel, theta_global::AbstractVector) =
    extract_submodel_params(melded.param_indices, theta_global)



end