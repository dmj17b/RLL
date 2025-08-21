import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import bipedal_walker


# Create a vectorized environment
vec_envs = make_vec_env(bipedal_walker.BipedalWalker, n_envs=16)

# Create a callback to save the model periodically
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='bipedal_walker_ppo')

# Initialize the PPO agent
# model = PPO('MlpPolicy', vec_envs, verbose=1, device='cpu', )
model = PPO.load("recent", env=vec_envs, device='cpu')


# Train the agent
model.learn(total_timesteps=50000000, callback=checkpoint_callback)

# Save the model
model.save("bipedal_walker_ppo")