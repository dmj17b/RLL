import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import lunar_lander

# Create a vectorized environment
vec_envs = make_vec_env(lunar_lander.LunarLander, n_envs=16)

# Initialize the PPO agent
model = PPO('MlpPolicy', vec_envs, verbose=1, device='cpu', )

# Train the agent
model.learn(total_timesteps=5000000)

# Save the model
model.save("lunar_lander_ppo")