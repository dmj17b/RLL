import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import lunar_lander
# Create a vectorized environment
env = make_vec_env(lunar_lander.LunarLander, n_envs=1, env_kwargs={'render_mode': 'human'})

# Load the trained PPO agent
model = PPO.load("lunar_lander_ppo", env=env, device='cpu')

# Evaluate the agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")  # Render the environment to visualize the agent's performance
    if dones:
        if env.envs[0].unwrapped.lander.awake:
            print("Lander successfully landed!")
        else:
            print("Lander crashed or left the viewport.")

