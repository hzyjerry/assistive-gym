from gym.envs.registration import register

tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch', 'Panda']

for task in tasks:
    for robot in robots:
        register(
            id='%s%s-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sEnv' % (task, robot),
            max_episode_steps=200,
        )

for task in ['ScratchItch', 'Feeding']:
    for robot in robots:
        register(
            id='%s%sMesh-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sMeshEnv' % (task, robot),
            max_episode_steps=200,
        )

register(
    id='HumanTesting-v1',
    entry_point='assistive_gym.envs:HumanTestingEnv',
    max_episode_steps=200,
)

register(
    id='SMPLXTesting-v1',
    entry_point='assistive_gym.envs:SMPLXTestingEnv',
    max_episode_steps=200,
)

register(
    id='SMPLXCreating-v1',
    entry_point='assistive_gym.envs:SMPLXCreatingEnv',
    max_episode_steps=1,
)


### Interactive enviornments

register(
    id='BedBathingJacoHumanPose-v1',
    entry_point='assistive_gym.envs:BedBathingJacoHumanPoseEnv',
    max_episode_steps=200,
)

for control_type in range(1, 5):
    register(
        id=f'BedBathingJacoHumanPose-v1{control_type}',
        entry_point='assistive_gym.envs:BedBathingJacoHumanPoseEnv',
        max_episode_steps=200,
        kwargs={
            "control_type": control_type
        }
    )

for control_type in range(1, 5):
    register(
        id=f'BedBathingJacoRobotPose-v1{control_type}',
        entry_point='assistive_gym.envs:BedBathingJacoRobotPoseEnv',
        max_episode_steps=200,
        kwargs={
            "control_type": control_type
        }
    )


### Interactive enviornments

register(
    id='ScratchItchJacoHumanPose-v1',
    entry_point='assistive_gym.envs:ScratchItchJacoHumanPoseEnv',
    max_episode_steps=200,
)

for control_type in range(1, 5):
    register(
        id=f'ScratchItchJacoHumanPose-v1{control_type}',
        entry_point='assistive_gym.envs:ScratchItchJacoHumanPoseEnv',
        max_episode_steps=200,
        kwargs={
            "control_type": control_type
        }
    )


for control_type in range(1, 5):
    register(
        id=f'ScratchItchJacoRobotPose-v1{control_type}',
        entry_point='assistive_gym.envs:ScratchItchJacoRobotPoseEnv',
        max_episode_steps=200,
        kwargs={
            "control_type": control_type
        }
    )



register(
    id='BedBathingJacoRobotPose-v1',
    entry_point='assistive_gym.envs:BedBathingJacoRobotPoseEnv',
    max_episode_steps=200,
)
