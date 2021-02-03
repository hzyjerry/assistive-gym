import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh

class SMPLXCreatingEnv(AssistiveEnv):
    def __init__(self):
        super(SMPLXCreatingEnv, self).__init__(robot=None, human=None, task='smplx_testing', obs_robot_len=0, obs_human_len=0)
        self._create_model_idx = 0

    def step(self, action):
        self.take_step(action, gains=0.05, forces=1.0)
        return [], 0, False, {}

    def _get_obs(self, agent=None):
        return []

    def reset(self):
        self._create_model_idx += 1
        # self._create_file_name = f"human_mesh_{self._create_model_idx:02d}.xyz"
        self._create_file_name = f"human_mesh_{self._create_model_idx:02d}.ply"

        super(SMPLXCreatingEnv, self).reset()

        ## Hacky: fix seed
        self.seed(0)

        self.build_assistive_env(furniture_type='wheelchair2')
        self.furniture.set_on_ground()

        self.human_mesh = HumanMesh()

        h = self.human_mesh
        body_shape = 'female_1.pkl'
        #body_shape = self.np_random.randn(1, self.human_mesh.num_body_shape)
        body_shape = self.np_random.randn(1, self.human_mesh.num_body_shape)
        joint_angles = [(self.human_mesh.j_left_hip_x, -90), (self.human_mesh.j_right_hip_x, -90), (self.human_mesh.j_left_knee_x, 70), (self.human_mesh.j_right_knee_x, 70), (self.human_mesh.j_left_shoulder_z, -45), (self.human_mesh.j_left_elbow_y, -90)]
        joint_angles += [(self.human_mesh.j_right_pecs_y, 0), (self.human_mesh.j_right_pecs_z, 0), (self.human_mesh.j_right_shoulder_x, 0), (self.human_mesh.j_right_shoulder_y, 0), (self.human_mesh.j_right_shoulder_z, -45 + self._create_model_idx * 5), (self.human_mesh.j_right_elbow_y, 45), (self.human_mesh.j_waist_x, 7.5), (self.human_mesh.j_waist_y, 0), (self.human_mesh.j_waist_z, 0)]


        self.human_mesh.init(self.directory, self.id, self.np_random, gender='female', height=1.7, body_shape=body_shape, joint_angles=joint_angles, position=[0, 0, 0], orientation=[0, 0, 0], save_name=self._create_file_name)

        # human_height, human_base_height = self.human_mesh.get_heights(set_on_ground=True)
        # print('Human height:', human_height, 'm')

        # self.human_mesh.set_base_pos_orient([0, -0.05, 1.0], [0, 0, 0, 1])
        chair_seat_position = np.array([0, 0.05, 0.6])
        self.human_mesh.set_base_pos_orient(chair_seat_position - self.human_mesh.get_vertex_positions(self.human_mesh.bottom_index), [0, 0, 0, 1])
        pos, orient = self.human_mesh.get_base_pos_orient()



        vertex_index = self.np_random.choice(self.human_mesh.right_arm_vertex_indices)
        target_pos = self.human_mesh.get_vertex_positions(vertex_index)
        self.target = self.create_sphere(radius=0.01, mass=0.0, pos=target_pos, visual=True, collision=False, rgba=[0, 1, 1, 1])

        p.setGravity(0, 0, 0, physicsClientId=self.id)
        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)


        self.init_env_variables()
        return self._get_obs()
