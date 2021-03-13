import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac
from ray.tune.logger import pretty_print
from numpngw import write_apng
from assistive_gym.utils import *


def train(env_name,
          algo,
          timesteps_total=1000000,
          save_dir='./trained_models/',
          policy_pair=None,
          coop=False,
          train_robot_only=False,
          seed=0,
          extra_configs={},
          save_per_iter=1,
          params={},
          save_env_name=None):
    """Training main function.main

    Params:
        coop: bool, if True, both human & robot actionable, else human static
        train_robot_only: bool, if True, freeze human policy and only train robot
        human_policy_path: str, xxx/checkpoint_xxx/checkpoint-xxx, if not None, load from prev checkpoint
        robot_policy_path: str, xxx/checkpoint_xxx/checkpoint-xxx, if not None, load from prev checkpoint
        save_env_name: str, save to {save_dir}/{algo}/{save_env_name}/

    """
    from torch.utils.tensorboard import SummaryWriter
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)

    # Sanity check
    if "Personalized" in env_name: assert coop and train_robot_only

    # Make environment & load checkpoints
    env = make_env(env_name, coop)
    agent, checkpoint_path = load_policy_pair(env, algo, env_name, policy_pair, coop=coop, seed=seed, train_robot_only=train_robot_only, extra_configs=extra_configs)
    env.disconnect()

    # Save
    save_env_name = env_name if save_env_name is None else save_env_name
    save_path = os.path.join(save_dir, algo, save_env_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving to {save_path}")
    with open(os.path.join(save_path, "params.yaml"), "w+") as f:
        yaml.dump(params, f)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs", save_env_name), comment=save_env_name)

    # Start Training
    timesteps = -1 # allow finetuning checkpoints, i.e. training 10M more steps from a 10M checkpoint
    while timesteps < timesteps_total:
        result = agent.train()
        if timesteps < 0:
            timesteps_total += result['timesteps_total']
        timesteps = result['timesteps_total']
        if coop:
            # Rewards are added in multi agent envs, so we divide by 2 since agents share the same reward in coop
            result['episode_reward_mean'] /= 2
            result['episode_reward_min'] /= 2
            result['episode_reward_max'] /= 2
        print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        sys.stdout.flush()

        iteration = result['training_iteration']
        writer.add_scalar("vals/fps", result['timesteps_total']/result['time_total_s'], iteration)
        writer.add_scalar("vals/meanReward", result['episode_reward_mean'], iteration)
        writer.add_scalar("vals/minReward", result['episode_reward_min'], iteration)
        writer.add_scalar("vals/maxReward", result['episode_reward_max'], iteration)
        writer.flush()
        # Delete the old saved policy
        if checkpoint_path is not None:
            shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        # Save the recently trained policy
        if result['training_iteration'] % save_per_iter == 0:
            checkpoint_path = agent.save(save_path)
    # Save at the end
    checkpoint_path = agent.save(save_path)
    writer.close()
    return checkpoint_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('params_file', default=None,
                        help='yaml parameter file')
    parser.add_argument('--cloud', action='store_true', default=False,
                        help='Cloud mode')
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
    assert params is not None

    env_name, algo, seed = params.env, params.algo, params.seed
    if args.cloud:
        params.save_dir = params.cloud_dir


    # Save to: save_dir/date
    save_dir = os.path.join(params.save_dir, str(params.date))
    print(f"Train on environment {env_name}")
    params.pprint()

    train(env_name,
          algo,
          timesteps_total=params.train_timesteps,
          save_dir=save_dir,
          policy_pair=params.policy_pair,
          coop=params.coop,
          train_robot_only=params.train_robot_only,
          seed=seed,
          save_per_iter=params.save_per_iter,
          params=params.toDict(),
          save_env_name=params.save_env_name)
