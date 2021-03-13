import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac
from ray.tune.logger import pretty_print
from numpngw import write_apng
from assistive_gym.utils import *


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


def render_policy(env_name, algo, policy_pair, coop=False, save_fig=False, seed=0, extra_configs={}, num_eps=1, image_path=".", save_env_name=None):
    """Training main function.main

    Params:
        coop: bool, if True, both human & robot actionable, else human static
        train_robot_only: bool, if True, freeze human policy and only train robot

    """

    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)

    env = make_env(env_name, coop)
    env.seed(seed)
    if save_fig:
        env.setup_camera(**camera_configs["BedBathing"])

    save_env_name = env_name if save_env_name is None else save_env_name
    test_agent, _ = load_policy_pair(env, algo, env_name, policy_pair, coop=coop, seed=seed, extra_configs=extra_configs)

    os.makedirs(image_path, exist_ok=True)
    if not save_fig:
        env.render()
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
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)
                rew = reward if not rew else rew + reward
            if save_fig:
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
        if save_fig:
            if coop:
                filename = f"{image_path}/{save_env_name}_{eps_i:02d}_rew_{rew['robot']:.03f}.png"
            else:
                filename = f"{image_path}/{save_env_name}_{eps_i:02d}_rew_{rew:.03f}.png"
            write_apng(filename, frames, delay=50)
            # filename = f'output_{env_name}_{eps_i:02d}.gif'
            # write_gif(frames, filename, fps=20)
            #return filename
        if coop:
            for key, val in rew.items():
                print(f"Reward {key}: {val}")
            else:
                print(f"Reward {rew}")
    env.disconnect()




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
    assert params is not None

    env_name, algo, seed = params.env, params.algo, params.seed

    # Save to: save_dir/date
    save_dir = os.path.join(params.save_dir, str(params.date))
    print(f"Render on environment {env_name}")
    params.pprint()

    image_path = os.path.join(params.image_dir, str(params.date), params.save_env_name)
    render_policy(env_name,
                  algo,
                  policy_pair=params.policy_pair,
                  save_fig=params.save_fig,
                  coop=params.coop,
                  seed=seed,
                  num_eps=params.render_eps,
                  image_path=image_path,
                  save_env_name=params.save_env_name)
