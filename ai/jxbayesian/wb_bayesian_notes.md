Unused material in wb_bayesian_slides.qmd

# Webinar goal

This webinar addresses two categories of people:

- Those who don't know anything about Bayesian statistics and use traditional frequentist probabilities
- Stan users

Its goal is to have the former group consider the Bayesian approach for their research and the second group consider looking into JAX

(the probability of a hypothesis in this theory is either $0$ or $1$—the event happens or it doesn't)

(for events for which long-runs don't make sense, one imagines alternative realities and considers the frequency of occurrences in those realities)

(a hypothesis is either true or false, so we can't speak of the probability of a hypothesis being true. Similarly, we can't speak of the probability of a parameter falling in a confidence interval—either it is in it or it is not)

JAX is designed natively for GPU coding and parallelism. Stan is not

[pymc-extras](https://github.com/pymc-devs/pymc-extras) is a testing ground for novel algorithms, distributions, and methods

## Annex {.center}

[List of currently available PPLs](https://en.wikipedia.org/wiki/Probabilistic_programming#List_of_probabilistic_programming_languages)

[statistical inference](https://en.wikipedia.org/wiki/Statistical_inference)
[probability distribution](https://en.wikipedia.org/wiki/Probability_distribution)

## Stan algorithms {.center}

gradient-based MCMC algorithms for Bayesian inference:
- Hamiltonian Monte Carlo (HMC)
- No-U-Turn sampler (NUTS), a variant of HMC and Stan's default MCMC engine

stochastic gradient-based variational algorithms for approximate Bayesian inference:
- Automatic Differentiation Variational Inference
- Pathfinder: Parallel quasi-Newton variational inference

gradient-based optimization algorithms for penalized maximum likelihood estimation:
- Limited-memory BFGS (Stan's default optimization algorithm)
- Broyden–Fletcher–Goldfarb–Shanno algorithm

- Laplace's approximation for classical standard error estimates and approximate Bayesian posteriors

https://bob-carpenter.github.io/stan-getting-started/stan-getting-started.html

https://en.wikipedia.org/wiki/Markov_chain

:::{.note}

For Julia users, there is [Turing.jl](https://github.com/TuringLang/Turing.jl)

:::

https://discourse.pymc.io/t/user-experience-python-vs-r-pymc-vs-stan-pytensor-vs-jax/16426/5

https://statmodeling.stat.columbia.edu/author/bob_carpenter/

https://www.pymc-labs.com/blog-posts/pymc-stan-benchmark/
https://elizavetasemenova.github.io/prob-epi/01_intro.html

NUTS: https://mc-stan.org/bayesplot/reference/MCMC-nuts.html

https://statswithr.github.io/book/stochastic-explorations-using-mcmc.html
https://pymc-devs.medium.com/the-future-of-pymc3-or-theano-is-dead-long-live-theano-d8005f8a0e9b
https://www.pymc-labs.com/blog-posts/pymc-stan-benchmark/
https://statmodeling.stat.columbia.edu/2024/09/25/stan-faster-than-jax-on-cpu/
