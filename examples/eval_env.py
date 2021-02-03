import gym, assistive_gym
from assistive_gym.learn import train, render_policy, evaluate_policy
import pybullet as p
import numpy as np
import gym

# Select the environment we want to test
env_name = 'BedBathingSawyerHuman-v1'
# Run the pretrained robot controller for 100 simulation trials to compute average reward/success
evaluate_policy(env_name, 'ppo', 'trained_models', n_episodes=100, seed=0)

