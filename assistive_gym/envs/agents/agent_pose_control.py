
from .human import Human
from .agent import Agent
from .robot import Robot
from .jaco import Jaco
import yaml
import numpy as np

class AgentPoseControl():

    def __init__(self, controllable_joint_indices, program_type=0, control_type=None, pose_file=None, *args):
        self.controllable_joint_indices = controllable_joint_indices
        self._control_types = []
        self._control_type = control_type
        self._control_poses = []
        self._scheduled_poses = []
        self._initial_pose = None

        self.initialize_program(program_type, pose_file=pose_file)
        self._t = 0

    def initialize_program(self, program_type, *args, **kwargs):
        pass

    @property
    def control_types(self):
        return self._control_types

    def _move_to_pose(self, next_pose, num_steps):
        """ Linearly move to next pose.
        """
        curr_pose = self.get_joint_angles(self.controllable_joint_indices)
        d_pose = (next_pose - curr_pose) / num_steps
        new_pose = curr_pose + d_pose
        self.set_joint_angles(self.controllable_joint_indices, new_pose)

    def control(self):
        for (trange, pose) in self._scheduled_poses:
            if trange[0] <= self._t and trange[1] > self._t:
                num_steps = trange[1] - trange[0]
                self._move_to_pose(pose, num_steps)
        self._t += 1

    def _reset_control_program(self):
        assert self._program_type is not None
        if self._program_type == 0:
            if self._control_type:
                control_type = self._control_type
            else:
                control_type = self.np_random.choice(self._control_types)
            self.control_program_v0(control_type)

    def reset(self):
        self._scheduled_poses = []
        self._reset_control_program()
        self._t = 0
        self._initial_pose = self.get_joint_angles(self.controllable_joint_indices)


class HumanPoseControl(AgentPoseControl, Human):

    def __init__(self, controllable_joint_indices, program_type=0, control_type=None, pose_file=None, controllable=False):
        Human.__init__(self, controllable_joint_indices, controllable=False)
        AgentPoseControl.__init__(self, controllable_joint_indices, program_type, control_type, pose_file)

    def initialize_program(self, program_type, pose_file):
        self._program_type = program_type
        if program_type == 0:
            self.init_program_v0(pose_file)

    def init_program_v0(self, pose_file):
        ## Load pose file
        self._control_types = range(1, 5)
        with open(pose_file) as f:
            self._control_poses = yaml.load(f)


    def control_program_v0(self, control_type=1):
        """ Controller v0: type 1~4

        Types:
            Type 1: static human (H 0%, R 100%)
            Type 2: human move to convenient location, stops (H 15%, R 85%)
            Type 3: human move to convenient location, moves n times (H 30%, R 50%)
            Type 4: human move to convenient location, moves n times (H 50%, R 50%)

        Note:
            Programmed as interpolating between poses
        """
        steps_per_pose = 50
        steps_transition = 10

        def _add_poses(nposes):
            for pi in range(nposes):
                rand_idx = self.np_random.choice(range(len(self._control_poses)))
                rand_pose = self._control_poses[rand_idx]
                start_t = pi * steps_per_pose
                end_t = pi * steps_per_pose + (pi + 1) * steps_transition
                self._scheduled_poses.append([(start_t, end_t), np.array(rand_pose)])

        if control_type == 1:
            # Do not move
            num_poses = 0
            _add_poses(num_poses)
        elif control_type == 2:
            # Move to a random pose and stop
            num_poses = 1
            _add_poses(num_poses)
        elif control_type == 3:
            # Move between a few random poses
            num_poses = 2
            _add_poses(num_poses)
        elif control_type == 4:
            # Move between a few random poses
            num_poses = 3
            _add_poses(num_poses)
        else:
            raise NotImplementedError(f"Human Pose Control Type {control_type} not implemented")

class RobotPoseControl(AgentPoseControl):
    pass

class JacoPoseControl(RobotPoseControl, Jaco):
    def __init__(self, controllable_joints='right', program_type=0, control_type=None, pose_file=None):
        Jaco.__init__(self, controllable_joints)
        AgentPoseControl.__init__(self, self.controllable_joint_indices, program_type, control_type, pose_file)
        self.controllable = False

    def initialize_program(self, program_type, pose_file):
        self._program_type = program_type
        if program_type == 0:
            self.init_program_v0(pose_file)

    def init_program_v0(self, pose_file):
        ## Load pose file
        self._control_types = range(1, 4)
        with open(pose_file) as f:
            self._control_poses = yaml.load(f)

    def control_program_v0(self, control_type=1):
        """ Controller v0: type 1~4

        Types:
            Type 1: static robot (R 0%, H 100%)
            Type 2: robot move to convenient location, stops (R 15%, H 85%)
            Type 3: robot move to convenient location, moves n times (R 30%, H 50%)

        Note:
            Programmed as interpolating between poses
        """
        assert control_type in range(1, 5)
        steps_per_pose = 50
        steps_transition = 10

        def _add_poses(nposes):
            for pi in range(nposes):
                rand_idx = self.np_random.choice(range(len(self._control_poses)))
                rand_pose = self._control_poses[rand_idx]
                start_t = pi * steps_per_pose
                end_t = pi * steps_per_pose + (pi + 1) * steps_transition
                self._scheduled_poses.append([(start_t, end_t), np.array(rand_pose)])

        if control_type == 1:
            # Do not move
            num_poses = 0
            _add_poses(num_poses)
        elif control_type == 2:
            # Move to a random pose and stop
            num_poses = 1
            _add_poses(num_poses)
        elif control_type == 3:
            # Move between a few random poses
            num_poses = 2
            _add_poses(num_poses)
        elif control_type == 4:
            # Move between a few random poses
            num_poses = 3
            _add_poses(num_poses)
        else:
            raise NotImplementedError(f"Robot Pose Control Type {control_type} not implemented")