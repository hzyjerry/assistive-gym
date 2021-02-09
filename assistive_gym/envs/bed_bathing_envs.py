from .bed_bathing import BedBathingEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from .agents.agent_pose_control import HumanPoseControl, JacoPoseControl

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'left'
human_controllable_joint_indices = human.right_arm_joints
class BedBathingPR2Env(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingBaxterEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingSawyerEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingJacoEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))




class BedBathingJacoHumanPoseEnv(BedBathingEnv):
    """Human using pose controller, train robot."""
    def __init__(self, control_type=None):
        super(BedBathingJacoHumanPoseEnv, self).__init__(robot=Jaco(robot_arm), human=HumanPoseControl(human_controllable_joint_indices, program_type=0, pose_file="poses/bed_bathing_human_poses.yaml", control_type=control_type))

class BedBathingJacoRobotPoseEnv(BedBathingEnv):
    """Robot using pose controller, train human."""
    def __init__(self, control_type=None):
        super(BedBathingJacoRobotPoseEnv, self).__init__(robot=JacoPoseControl(robot_arm, program_type=0, pose_file="poses/bed_bathing_jaco_poses.yaml", control_type=control_type), human=Human(human_controllable_joint_indices, controllable=True))





class BedBathingStretchEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingPandaEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class BedBathingPR2HumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:BedBathingPR2Human-v1', lambda config: BedBathingPR2HumanEnv())

class BedBathingBaxterHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:BedBathingBaxterHuman-v1', lambda config: BedBathingBaxterHumanEnv())

class BedBathingSawyerHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:BedBathingSawyerHuman-v1', lambda config: BedBathingSawyerHumanEnv())

class BedBathingJacoHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self, frame_skip=5, action_multiplier=0.05, collab_version='v4'):
        super(BedBathingJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True), frame_skip=frame_skip, action_multiplier=action_multiplier, collab_version=collab_version)
register_env('assistive_gym:BedBathingJacoHuman-v1', lambda config: BedBathingJacoHumanEnv())
register_env('assistive_gym:BedBathingJacoHuman-v1-skip1', lambda config: BedBathingJacoHumanEnv(frame_skip=1, action_multiplier=0.25))
register_env('assistive_gym:BedBathingJacoHuman-v1-skip5', lambda config: BedBathingJacoHumanEnv(frame_skip=5, action_multiplier=0.05))
register_env('assistive_gym:BedBathingJacoHuman-v1-skip5-collab3', lambda config: BedBathingJacoHumanEnv(frame_skip=5, action_multiplier=0.05, collab_version="v3"))
register_env('assistive_gym:BedBathingJacoHuman-v1-skip5-collab2', lambda config: BedBathingJacoHumanEnv(frame_skip=5, action_multiplier=0.05, collab_version="v2"))
register_env('assistive_gym:BedBathingJacoHuman-v1-skip5-collab1', lambda config: BedBathingJacoHumanEnv(frame_skip=5, action_multiplier=0.05, collab_version="v1"))

class BedBathingStretchHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:BedBathingStretchHuman-v1', lambda config: BedBathingStretchHumanEnv())

class BedBathingPandaHumanEnv(BedBathingEnv, MultiAgentEnv):
    def __init__(self):
        super(BedBathingPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:BedBathingPandaHuman-v1', lambda config: BedBathingPandaHumanEnv())

