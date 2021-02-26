""" Personalized environment dispatcher.

Notes:
- 2021.02.23 Train robot agent with 0%~100% human agents
"""

import os

def get_personalized_env_v0(env_name, load_root="human_models", checkpoint_path="checkpoint_520"):
    """
    Get environment collab version and saved path.

    2021.02.23 Note
    - env_name: BedBathingJacoHuman-v0217_0-v1
      output: human_models/ppo/BedBathingJacoHuman-v0217_h1-v1/checkpoint_520/checkpoint-520

    """
    collab_version = env_name.split("-")[1].replace("-v1", "")
    # human_models/ppo/BedBathingJacoHuman-v0217_h1-v1/checkpoint_520
    load_env_name = env_name.replace("Personalized", "Human")
    load_path = f"{load_root}/ppo/{load_env_name}/{checkpoint_path}/{checkpoint_path.replace('_', '-')}"

    assert os.path.isfile(load_path), f"Personalized policy not found: {load_path}"
    return load_path