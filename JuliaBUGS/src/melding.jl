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

import LogDensityProblems
using JuliaBUGS: model_parameters, BUGSModel

using Distributions
using AbstractPPL: VarName

export meld, MeldedModel
export AbstractPooling, ProductOfExperts, Dictatorial

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

"""
    meld(
    submodels::AbstractVector,
    links::AbstractVector,
    pooling::AbstractPooling
)

Meld the submodels in preparation for sampling. 
The user should provide a Vector of models, a Vector of symbols identifying the linking
parameter for each model, and a pooling method.
"""
function meld(
    submodels::AbstractVector,
    links::AbstractVector,
    pooling::AbstractPooling
)
    M = length(submodels)
    M >= 1 || throw(ArgumentError("There should be at least one submodel for melding"))

    length(links) == M || throw(ArgumentError("got $(length(links)) link specifications for $M submodels"))

    models = BUGSModel[m for m in submodels]

    priors = extract_prior.(models, links)

    param_indices, total_dims = build_param_map(submodels, links)

    return MeldedModel(
        submodels,
        links,
        priors,
        pooling,
        collect(param_indices),
        total_dims
    )
end

"""
    adjust_prior_logdensity(::ProductOfExperts, priors, phi_val)

Product of Experts adjustment to logdensity probability for a sample.
In this case no adjustment needed as the PoE pooling and adjustment cancel each other out.
"""
function adjust_prior_logdensity(::ProductOfExperts, priors, phi_val)
    return 0.0
end


"""
    adjust_prior_logdensity(p::Dictatorial, priors, phi_val)

Dictatorial adjustment to logdensity probability for a sample.
In this case the logdensity for each of the non-dicattorial priors needs to be
subtracted to prevent double counting.
"""
function adjust_prior_logdensity(p::Dictatorial, priors, phi_val)
    adj = 0.0
    for i in 1:length(priors)
        if i != p.dictator
            adj -= Distributions.logpdf(priors[i], phi_val)
        end
    end
    return adj
end

# Set the total count of unique parameters across all submodels
LogDensityProblems.dimension(melded::MeldedModel) = melded.dimension

# Declare basic log-density capability (LogDensityOrder{0} means target evaluation only)
LogDensityProblems.capabilities(::Type{<:MeldedModel}) = LogDensityProblems.LogDensityOrder{0}()

"""
    LogDensityProblems.logdensity(melded::MeldedModel, theta_global)

Evaluate the total log-density for a the melded model 
"""
function LogDensityProblems.logdensity(melded::MeldedModel, theta_global)

    phi_val = theta_global[1]

    submodel_thetas = extract_submodel_params(melded, theta_global)

    submodel_lps = LogDensityProblems.logdensity.(melded.submodels, submodel_thetas)
    base_lp = sum(submodel_lps)

    prior_adj = adjust_prior_logdensity(melded.pooling, melded.priors, phi_val)

    return base_lp + prior_adj
end

end