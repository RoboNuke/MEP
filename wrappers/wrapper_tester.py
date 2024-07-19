import gymnasium as gym
import mani_skill.envs
import torch
import time

from wrappers.manip_ellipsoid_wrapper import ManipulabilityEllipsoidObservationWrapper

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenObservationWrapper, FlattenRGBDObservationWrapper
num_env = 2

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_env,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    #parallel_gui_render_enabled=True,
    render_mode="sensors"
)
env = ManipulabilityEllipsoidObservationWrapper(env)
#print("Observation space", env.observation_space)
#print("Action Space:", env.action_space)
obs, _ = env.reset()
print("\nTesting flat state reset() obs")
assert obs.shape == (num_env, 55)
print("\tPassed")
next_obs, reward, terminations, truncations, infos = env.step(env.action_space.sample())
print("Testing flat state step() obs")
assert next_obs.shape == (num_env, 55)
print("\tPassed")
print("MEP Wrapper works for obs_mode='state'\n")

# test for state_dict
num_env = 2

from typing import Dict

def testMEPObs(obs, txt):
    global num_env
    print(txt)
    assert "MEP" in obs.keys()  
    mep_keys = obs['MEP'].keys()
    assert "volume" in mep_keys
    assert obs['MEP']['volume'].shape == (num_env, 1)
    assert "major_axis" in mep_keys
    assert obs['MEP']['major_axis'].shape == (num_env, 6)
    assert "minor_axis" in mep_keys
    assert obs['MEP']['major_axis'].shape == (num_env, 6)
    print("\tPassed!")



env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_env,
    obs_mode="state_dict", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    #parallel_gui_render_enabled=True,
    render_mode="sensors"
)
env = ManipulabilityEllipsoidObservationWrapper(env)
#print("Observation space", env.observation_space)
#print("Action Space:", env.action_space)
obs, _ = env.reset()

testMEPObs(obs,"Testing state_dict reset() obs...")

next_obs, reward, terminations, truncations, infos = env.step(env.action_space.sample())

testMEPObs(obs,"Testing state_dict step() Obs...")
print("MEP Wrapper works for obs_mode='state_dict'\n")


# test for rgb state
num_env = 2

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_env,
    obs_mode="rgbd", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    #parallel_gui_render_enabled=True,
    render_mode="sensors"
)
env = FlattenRGBDObservationWrapper(env, rgb_only=True)
env = ManipulabilityEllipsoidObservationWrapper(env)

reset_obs, _ = env.reset()
print("\nTesting flattened rgb reset() obs")
assert reset_obs['state'].shape == (num_env, 42)
print("\tPassed")
step_obs, reward, terminations, truncations, infos = env.step(env.action_space.sample())
print("Testing flattened rgb step() obs")
assert step_obs['state'].shape == (num_env, 42)
print("\tPassed")

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_env,
    obs_mode="rgbd", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    #parallel_gui_render_enabled=True,
    render_mode="sensors"
)
env = FlattenRGBDObservationWrapper(env, rgb_only=False)
env = ManipulabilityEllipsoidObservationWrapper(env)
reset_obs, _ = env.reset()
print("\nTesting flattened rgbd reset() obs")
assert reset_obs['state'].shape == (num_env, 42)
print("\tPassed")
step_obs, reward, terminations, truncations, infos = env.step(env.action_space.sample())
print("Testing flattened rgbd step() obs")
assert step_obs['state'].shape == (num_env, 42)
print("\tPassed")
print("MEP Wrapper works for flattened obs_mode='rgbd'\n")

# test for rgb state
num_env = 2

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=num_env,
    obs_mode="rgbd", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    #parallel_gui_render_enabled=True,
    render_mode="sensors"
)

env = ManipulabilityEllipsoidObservationWrapper(env)

reset_obs, _ = env.reset()
testMEPObs(reset_obs, "Testing rgbd reset() obs...")
step_obs, _,_,_,_ = env.step(env.action_space.sample())
testMEPObs(step_obs, "Testing rgbd step() obs...")
print("MEP Wrapper works for obs_mode='rgbd'\n")

print("Observation Space Values:")
obs = env.observation_space
for pkey in obs.keys():
    print(pkey)
    for ckey in obs[pkey].keys():
        print("\t", ckey)