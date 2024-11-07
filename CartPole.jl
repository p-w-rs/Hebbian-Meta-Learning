using PyCall, BlackBoxOptim

include("NNLib.jl")
using .NNLib

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

function episode(env, params)
    model = restructure(params)
    rewards = zeros(10)
    for i in 1:10
        (observation, info), done = env.reset(seed=i), false
        total_reward = 0.0
        while !done
            action = argmax(model(observation))
            observation, reward, terminated, truncated, info = env.step(action - 1)
            done = terminated || truncated
            total_reward += reward
        end
        rewards[i] = total_reward
    end
    return -sum(rewards) / 10
end

res = bboptimize(
    ps -> episode(env, Float32.(ps)); Method=:adaptive_de_rand_1_bin_radiuslimited,
    SearchRange=(-1.0, 1.0), NumDimensions=length(x0),
    MaxSteps=50000, TraceMode=:compact
);

fitness = best_fitness(res)
params = Float32.(best_candidate(res))
env = gym.make("CartPole-v1", render_mode="human")
total_reward = episode(env, params)
println("Total reward: $total_reward")
