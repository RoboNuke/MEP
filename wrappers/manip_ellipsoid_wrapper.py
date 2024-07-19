import gymnasium as gym
from gymnasium import spaces
import mani_skill.envs
from mani_skill.utils import common
from mani_skill.envs.sapien_env import BaseEnv

import sapien.physx as physx
from typing import Dict
import torch

class ManipulabilityEllipsoidObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        # assert control is pd_ee_pose
        # assert obs is still a dict
        self.base_env: BaseEnv = env.unwrapped
        wrapped_observation_space = env.observation_space

        # check it is a dict
        #if not isinstance(wrapped_observation_space, spaces.Dict):
        #    raise ValueError(
        #        f"ManipulabilityEllipsoidObservationWrapper is only usable with dict observations, "
        #        f"environment observation space is {type(wrapped_observation_space)}"
        #    )
        self.is_dict = isinstance(wrapped_observation_space, spaces.Dict)
        # check controller type is correct
        if( not 'pd_ee_delta_pose' in self.base_env.agent.controllers.keys() or 
            not 'arm' in self.base_env.agent.controllers['pd_ee_delta_pose'].controllers.keys()):
            raise ValueError(
                f"ManipulabilityEllipsoidObservationWrapper is only usable with pd_ee_delta_pose controller type, "
                f"environment controller types are {self.base_env.agent.controllers.keys()}"
            )
        
        self.base_env.update_obs_space(
            common.to_numpy(
                self.observation(common.to_tensor(self.base_env._init_raw_obs))
            )
        )
        super().__init__(env)

    def reset(self, **kwargs):
        #self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs: Dict):
        con = self.base_env.agent.controllers['pd_ee_delta_pose'].controllers['arm']
        if( physx.is_gpu_enabled()):
            jacobian = (
                    con.fast_kinematics_model.jacobian_mixed_frame_pytorch(
                        con.articulation.get_qpos()[:, con.active_ancestor_joint_idxs]
                    )
                    .view(-1, len(con.active_ancestor_joints), 6)
                    .permute(0, 2, 1)
                )
            U, S, Vt = torch.linalg.svd(jacobian)
            # note columns of U are the eigen vectors of J J^T 
            volume = torch.prod(S, 1).reshape(self.base_env.num_envs,1) # multiply along the first axis to approx volume
            majAxis = torch.flatten(U[:, 0, :], 1)
            minAxis = torch.flatten(U[:, 1, :], 1)


            
        else:
            # typing here doesn't seem right (not a torch but numpy still)
            jacobian = (con.pmodel.compute_single_link_local_jacobian(
                    common.to_numpy(con.articulation.get_qpos()).squeeze(0),
                    9
                )
                .reshape(1, 6, 9)
            )
            raise NotImplementedError
        if self.is_dict and len(obs.keys()) == 2 and ('rgbd' in obs.keys() or 'rgb' in obs.keys()) and 'state' in obs.keys():
            obs['state'] = torch.cat((obs['state'], volume, majAxis, minAxis), 1)
        elif self.is_dict:
            obs['MEP'] = {}
            obs['MEP']['volume'] = volume
            obs['MEP']['major_axis'] = majAxis
            obs['MEP']['minor_axis'] = minAxis
        else:
            obs = torch.cat((obs, volume, majAxis, minAxis), 1)
        return obs
