import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac
from ray.tune.logger import pretty_print
from numpngw import write_apng
from assistive_gym.utils import *


def evaluate_policy(env_name, algo, human_policy_path=None, robot_policy_path=None, n_episodes=100, coop=False, seed=0, verbose=False, extra_configs={}, log_info=False):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop)

    test_agent, _ = load_policy_pair(env, algo, env_name, human_policy_path, robot_policy_path, coop=coop, seed=seed, train_robot_only=train_robot_only, extra_configs=extra_configs)

    rewards = []
    forces = []
    task_successes = []
    eval_info = {}
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        force_list = []
        task_success = 0.0
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                reward = reward['robot']
                done = done['__all__']
                info = info['robot']
            else:
                action = test_agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
            reward_total += reward
            force_list.append(info['total_force_on_human'])
            task_success = info['task_success']
            if log_info:
                for key, val in info.items():
                    if key not in eval_info:
                        eval_info[key] = [val]
                    else:
                        eval_info[key].append(val)

        rewards.append(reward_total)
        forces.append(np.mean(force_list))
        task_successes.append(task_success)
        if verbose:
            print(f'({episode}/{n_episodes}) Reward total: {reward_total:.2f}, mean force: {np.mean(force_list):.2f}, task success: {task_success}')
            print(f'({episode}/{n_episodes}) Eval info')
            for key, val_list in eval_info.items():
                print(f'\t{key} mean: {np.mean(val_list):.03f} std: {np.std(val_list):.03f}')
        sys.stdout.flush()
    env.disconnect()

    print('\n', '-'*50, '\n')
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    # print('Forces:', forces)
    print('Force Mean:', np.mean(forces))
    print('Force Std:', np.std(forces))

    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes))
    print('Task Success Std:', np.std(task_successes))
    sys.stdout.flush()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('params_file', default=None,
                        help='yaml parameter file')
    args = parser.parse_args()

    """Arguments:
    Mode:
        common: env, algo, seed
        train_args:
            train-timesteps, save-dir, save-per-iter, load-policy-path
        eval_args
            load-policy-path, eval-episodes
        render_args
            load-policy-path, eval-episodes, save_fig
    """

    from dotmap import DotMap
    import yaml

    params = None
    with open(args.params_file, "r") as f:
        params = DotMap(yaml.load(f))


    checkpoint_path = None
    env_name, algo, seed = params.env, params.algo, params.seed

    print(f"Eval on environment {env_name}")
    evaluate_policy(env_name,
                    algo,
                    human_policy_path=params.human_policy_path,
                    robot_policy_path=params.robot_policy_path,
                    n_episodes=params.eval_episodes,
                    coop=coop,
                    seed=seed,
                    verbose=params.verbose,
                    log_info=params.log_info)
    params.pprint()
