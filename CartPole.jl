# Main script (LunarLander.jl)
using PyCall, StatsBase
using BSON: @save

include("NNLib.jl")
using .NNLib

include("ESOptimizer.jl")
using .ESOptimizer

# Load the environment
gym = pyimport("gymnasium")
env = gym.make("CartPole-v1")
obs_space = env.observation_space.shape[1]
action_space = 2

# Define the model
model = Chain(
    Dense(obs_space => 64, tanh),
    Dense(64 => 32, tanh),
    Dense(32 => 16, tanh),
    Dense(16 => 8, tanh),
    Dense(8 => action_space)
)
x0, restructure = destructure(model)

function objective_fn(env, n, params)
    model = restructure(params)
    rewards = zeros(Float32, n)
    for i in 1:n
        (observation, info), done = env.reset(seed=i), false
        total_reward = 0.0f0
        while !done
            action = argmax(model(observation))
            observation, reward, terminated, truncated, info = env.step(action - 1)
            done = terminated || truncated
            total_reward += reward
        end
        rewards[i] = total_reward
    end
    return mean(rewards)
end

# Configure ES optimization
config = ESConfig(
    population_size=500,
    learning_rate=0.1f0,
    sigma=0.1f0,
    lr_decay=0.995f0,
    sigma_decay=0.999f0
)

# Run optimization
fitness_fn = params -> objective_fn(env, 20, params)
# Example usage with fitness cap
best_params, best_fitness = optimize(
    fitness_fn, Float32.(x0), config,
    generations=1000,
    seed=1,
    fitness_cap=500.0f0
)


# Test the best solution
env = gym.make("CartPole-v1", render_mode="human")
total_reward = objective_fn(env, 1, best_params)
println("Final test reward: $total_reward")
@save "results/cartpole.bson" best_params
