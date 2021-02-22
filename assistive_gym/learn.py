import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac
from ray.tune.logger import pretty_print
from numpngw import write_apng


camera_configs = {
    "BedBathing": {
        "camera_eye": [-0.7, -0.75, 1.5],
        "camera_target": [0.0, 0, 0.75],
        "camera_width": 1920//4,
        "fov": 60,
        "camera_height": 1080//4
    }
    # env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
}


def setup_config(env, algo, coop=False, seed=0, extra_configs={}):
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
    return {**config, **extra_configs}

def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'sac':
        agent = sac.SACTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
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

def make_env(env_name, coop=False):
    if not coop:
        return gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        return env_class()

def train(env_name, algo, timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, extra_configs={}, save_per_iter=1, params={}):
    from torch.utils.tensorboard import SummaryWriter
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop)
    agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed, extra_configs)
    env.disconnect()

    save_path = os.path.join(save_dir, algo, env_name)
    print(f"Saving to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "params.yaml"), "w+") as f:
        yaml.dump(params, f)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs", env_name), comment=env_name)

    timesteps = 0
    while timesteps < timesteps_total:
        result = agent.train()
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

def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, extra_configs={}, num_eps=1, image_path="."):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    os.makedirs(image_path, exist_ok=True)
    if not colab:
        env.render()
    # import pdb; pdb.set_trace()
    for eps_i in range(num_eps):
        print(f"Render eps {eps_i}")
        obs = env.reset()
        frames = []
        done = False
        rew = None
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                done = done['__all__']
                rew = reward if not rew else {key: rew[key] + reward[key] for key in reward.keys()}
                # print(f"Task reward {info['robot']['task_reward']:.04f} action reward {info['robot']['action_reward']:.04f}, preference reward {info['robot']['preference_reward']:.04f}")
                # import pdb; pdb.set_trace()
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)
                rew = reward if not rew else rew + reward
            if colab:
                def _resize(img, ratio):
                    import cv2
                    old_w, old_h = img.shape[1], img.shape[0]
                    new_w = int(old_w * ratio)
                    new_h = int(old_h * ratio)
                    img = cv2.resize(img, (new_w, new_h))
                    return img
                # Capture (render) an image from the camera
                img, depth = env.get_camera_image_depth()
                img = _resize(img, 0.5)
                frames.append(img)
        if colab:
            if coop:
                filename = f"{image_path}/output_{env_name}_{eps_i:02d}_rew_{rew['robot']:.03f}.png"
            else:
                filename = f"{image_path}/output_{env_name}_{eps_i:02d}_rew_{rew:.03f}.png"
            write_apng(filename, frames, delay=50)
            # filename = f'output_{env_name}_{eps_i:02d}.gif'
            # import pdb; pdb.set_trace()
            # write_gif(frames, filename, fps=20)
            #return filename
        if coop:
            for key, val in rew.items():
                print(f"Reward {key}: {val}")
            else:
                print(f"Reward {rew}")
    env.disconnect()


def evaluate_policy(env_name, algo, policy_path, n_episodes=100, coop=False, seed=0, verbose=False, extra_configs={}, log_info=False):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

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
    parser.add_argument('--cloud', action='store_true', default=False,
                        help='Cloud mode')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Whether to evaluate a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--launch', action='store_true', default=False,
                        help='Whether to launch')
    args = parser.parse_args()

    """Arguments:
    Mode:
        common: env, algo, seed
        train_args:
            train-timesteps, save-dir, save-per-iter, load-policy-path
        eval_args
            load-policy-path, eval-episodes
        render_args
            load-policy-path, eval-episodes, colab
    """

    from dotmap import DotMap
    import yaml

    params = None
    with open(args.params_file, "r") as f:
        params = DotMap(yaml.load(f))


    checkpoint_path = None
    env_name, algo, seed = params.env, params.algo, params.seed
    coop = ('Human' in env_name) and ('Pose' not in env_name)

    if args.cloud:
        params.save_dir = params.cloud_dir

    # Save to: save_dir/date
    save_dir = os.path.join(params.save_dir, str(params.date))
    load_policy_path = os.path.join(params.load_policy_path, str(params.date))

    if args.train:
        p = params.train
        print(f"Train on environment {env_name}")
        params.pprint()
        checkpoint_path = train(env_name, algo, timesteps_total=p.train_timesteps, save_dir=save_dir, load_policy_path=load_policy_path, coop=coop, seed=seed, save_per_iter=p.save_per_iter, params=params.toDict())

    if args.render:
        p = params.render
        env = make_env(env_name, coop)
        env.seed(seed)
        image_path = os.path.join(p.image_dir, str(params.date))
        if p.colab:
            env.setup_camera(**camera_configs["BedBathing"])
        render_policy(env, env_name, algo, checkpoint_path if checkpoint_path is not None else load_policy_path, coop=coop, colab=p.colab, seed=seed, num_eps=p.render_eps, image_path=image_path)

    if args.eval:
        p = params.eval
        evaluate_policy(env_name, algo, checkpoint_path if checkpoint_path is not None else load_policy_path, n_episodes=p.eval_episodes, coop=coop, seed=seed, verbose=p.verbose, log_info=p.log_info)

