module ESOptimizer

using Statistics, Random
using Distributions: Normal
using ProgressMeter

export optimize, ESConfig

struct ESConfig
    population_size::Int
    learning_rate::Float32
    sigma::Float32
    lr_decay::Float32
    sigma_decay::Float32
end

function ESConfig(;
    population_size=500,
    learning_rate=0.1f0,
    sigma=0.1f0,
    lr_decay=0.995f0,
    sigma_decay=0.999f0
)
    ESConfig(population_size, learning_rate, sigma, lr_decay, sigma_decay)
end

function optimize(f::Function, x0::Vector{Float32}, config::ESConfig;
    generations=1000, seed=nothing, fitness_cap=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    n_params = length(x0)
    θ = copy(x0)

    # Initialize adaptive parameters
    α = config.learning_rate
    σ = config.sigma

    best_fitness = -Inf32
    best_params = copy(θ)

    # Create normal distribution for sampling
    noise_dist = Normal(0.0f0, 1.0f0)

    # Progress tracking
    progress = Progress(generations, desc="Optimizing: ")

    for gen in 1:generations
        # Generate population of noise vectors
        ϵ = rand(noise_dist, n_params, config.population_size)

        # Evaluate fitness for each noise vector
        rewards = zeros(Float32, config.population_size)
        for i in 1:config.population_size
            noise = @view ϵ[:, i]
            candidate = θ .+ σ .* noise
            rewards[i] = f(candidate)
        end

        # Normalize rewards with numerical stability
        rewards_mean = mean(rewards)
        rewards_std = std(rewards)
        if rewards_std > 0
            rewards = (rewards .- rewards_mean) ./ (rewards_std + 1.0f-8)
        end

        # Compute weighted average of noise vectors
        update = zeros(Float32, n_params)
        for i in 1:config.population_size
            update .+= rewards[i] .* @view ϵ[:, i]
        end
        update .*= 1.0f0 / (config.population_size * σ)

        # Update parameters
        θ .+= α .* update

        # Update current best if better
        current_fitness = maximum(rewards) * rewards_std + rewards_mean
        if current_fitness > best_fitness
            best_fitness = current_fitness
            best_params .= θ
        end

        # Check if we've reached the fitness cap
        if !isnothing(fitness_cap) && best_fitness >= fitness_cap
            next!(progress, showvalues=[(:generation, gen),
                (:best_fitness, best_fitness),
                (:learning_rate, α),
                (:sigma, σ),
                (:status, "Fitness cap reached")])
            break
        end

        # Decay learning rate and noise standard deviation
        α *= config.lr_decay
        σ *= config.sigma_decay

        # Update progress
        next!(progress, showvalues=[(:generation, gen),
            (:best_fitness, best_fitness),
            (:learning_rate, α),
            (:sigma, σ)])
    end

    return best_params, best_fitness
end

end # module
