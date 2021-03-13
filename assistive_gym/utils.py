import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac
from ray.tune.logger import pretty_print
from numpngw import write_apng
from assistive_gym.env_dispatch import *


def setup_config(env, algo, coop=False, seed=0, train_robot_only=False, extra_configs={}):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [100, 100]
    elif algo == 'sac':
        config = sac.DEFAULT_CONFIG.copy()
    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    if algo == 'sac':
        config['timesteps_per_iteration'] = 400
        config['learning_starts'] = 1000
    if coop:
        obs = env.reset()
        policies = {'robot': (None, env.observation_space_robot, env.action_space_robot, {}), 'human': (None, env.observation_space_human, env.action_space_human, {})}
        config['multiagent'] = {'policies': policies, 'policy_mapping_fn': lambda a: a}
        config['env_config'] = {'num_agents': 2}
        if train_robot_only:
            config['multiagent']['policies_to_train'] = ["robot"]
    return {**config, **extra_configs}

def load_policy(env, algo, env_name, coop=False, policy_path=None, seed=0, train_robot_only=False, extra_configs={}):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, coop, seed, train_robot_only, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'sac':
        agent = sac.SACTrainer(setup_config(env, algo, coop, seed, train_robot_only, extra_configs), 'assistive_gym:'+env_name)
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:

            if algo not in policy_path:
                # Find the most recent policy in the directory
                # print(f"Policy path {policy_path} algo {algo} env name {env_name}")
                directory = os.path.join(policy_path, algo, env_name)
            else:
                directory = policy_path
            files = [int(f.split('_')[-1]) for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            if files:
                checkpoint_num = max(files)
                checkpoint_path = os.path.join(directory, 'checkpoint_%d' % checkpoint_num, 'checkpoint-%d' % checkpoint_num)
                print(f"Load from checkpoint {checkpoint_path}")
                agent.restore(checkpoint_path)
                # return agent, checkpoint_path
            else:
                print("Did not find checkpoint files")
            return agent, None
    return agent, None


def load_policy_pair(env, algo, env_name, policy_pair, coop=False, seed=0, train_robot_only=False, extra_configs={}):
    """Load human & robot policy pair.

    Params:
        policy_pari: dict, has fields
            human_policy_path: str
            human_policy_coop: bool
            robot_policy_path: str
            robot_policy_coop: bool
    """


    def get_agent(coop):
        if algo == 'ppo':
            agent = ppo.PPOTrainer(setup_config(env, algo, coop, seed, train_robot_only, extra_configs), 'assistive_gym:'+env_name)
        elif algo == 'sac':
            agent = sac.SACTrainer(setup_config(env, algo, coop, seed, train_robot_only, extra_configs), 'assistive_gym:'+env_name)
        return agent

    agent = get_agent(coop)

    if coop:
        # Init params
        human_params = agent.get_weights()['human']
        robot_params = agent.get_weights()['robot']

        human_policy_path = policy_pair.human_policy_path
        robot_policy_path = policy_pair.robot_policy_path
        if robot_policy_path == "same":
            robot_policy_path = human_policy_path

        if human_policy_path and 'checkpoint' in human_policy_path:
            print(f"Load human policy from {human_policy_path}")
            human_coop = policy_pair.human_policy_coop
            agent_ = get_agent(human_coop)
            agent_.restore(human_policy_path)
            if human_coop:
                human_params = agent_.get_weights()['human']
            else:
                human_params = agent_.get_weights()

        if robot_policy_path and 'checkpoint' in robot_policy_path:
            print(f"Load robot policy from {robot_policy_path}")
            robot_coop = policy_pair.robot_policy_coop
            agent_ = get_agent(robot_coop)
            agent_.restore(robot_policy_path)
            if robot_coop:
                robot_params = agent_.get_weights()['robot']
            else:
                robot_params = agent_.get_weights()

        agent.set_weights({'human': human_params, 'robot': robot_params})
    return agent, None

def make_env(env_name, coop=False):
    if not coop:
        return gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        return env_class()