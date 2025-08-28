import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import bipedal_walker
# Create a vectorized environment
env = make_vec_env(bipedal_walker.BipedalWalker, n_envs=1, env_kwargs={'render_mode': 'human', 'hardcore': True})

# Load the trained PPO agent
model = PPO.load("bipedal_walker_ppo", env=env, device='cpu')

# Evaluate the agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")  # Render the environment to visualize the agent's performance
