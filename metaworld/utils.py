import math
import os
import random
from collections import deque

import numpy as np
import scipy.linalg as sp_la

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.util.shape import view_as_windows
from torch import distributions as pyd

ALL_ENV_NAMES = ['assembly', 'basketball', 'bin-picking', 'box-close', 'button-press', 'button-press-wall', 'button-press-topdown', 'button-press-topdown-wall', 'dial-turn', 'disassemble', 'coffee-button', 'coffee-pull', 'coffee-push', 'door-lock', 'door-unlock', 'door-open', 'door-close', 'drawer-open', 'drawer-close', 'faucet-open', 'faucet-close', 'hammer', 'hand-insert', 'handle-press', 'handle-pull', 'handle-press-side', 'handle-pull-side', 'lever-pull', 'peg-insert-side', 'peg-unplug-side', 'pick-out-of-hole', 'pick-place', 'pick-place-wall', 'plate-slide', 'plate-slide-back', 'plate-slide-side', 'plate-slide-back-side', 'push', 'push-back', 'push-wall', 'reach', 'reach-wall', 'shelf-place', 'soccer', 'stick-push', 'stick-pull', 'sweep', 'sweep-into', 'window-open', 'window-close']

def obs2state(obs, action, env, env_name):
    # if env_name in ['door-open', 'door-close']:
    #     # eef_pos 0:3, eef_close_amount 3:4, obj_pos 4:7, obj_quat 7:11, eef_to_obj_pos 11:14, door_joint 14:15
    #     eef_pos = obs[0:3]
    #     eef_close_amount = obs[3:4]
    #     obj_pos = obs[4:7]
    #     obj_quat = obs[7:11]
    #     eef_to_obj_pos = np.array(eef_pos) - np.array(obj_pos)
    #     door_joint = [float(env.data.joint("doorjoint").qpos.item())]
    #     state = np.concatenate([eef_pos, eef_close_amount, obj_pos, obj_quat, eef_to_obj_pos, door_joint])
    
    if env_name in ALL_ENV_NAMES:
        # eef_pos 0:3, eef_close_amount 3:4, obj_pos 4:7, obj_quat 7:11, eef_to_obj_pos 11:14, target_pos 14:17, obj_to_target_pos 17:20
        eef_pos = obs[0:3]
        eef_close_amount = obs[3:4]
        obj_pos = obs[4:7]
        if env_name in ['reach', 'reach-wall']:
            obj_pos = eef_pos
        obj_quat = obs[7:11]
        eef_to_obj_pos = np.array(eef_pos) - np.array(obj_pos)
        target_pos = env._target_pos
        obj_to_target_pos = np.array(obj_pos) - np.array(target_pos)
        if env_name in ['button-press', 'button-press-wall', 'button-press-topdown', 'button-press-topdown-wall', 'coffee-button', 'door-open', 'door-close', 'drawer-open', 'drawer-close', 'reach', 'reach-wall', 'window-open', 'window-close']:
            check_grasp = 0
        else:
            check_grasp = env._gripper_caging_reward(
                action = action,
                obj_pos = obj_pos,
                object_reach_radius=0.04,
                obj_radius=0.02,
                pad_success_thresh=0.05,
                xz_thresh=0.05,
                desired_gripper_effort=0.7,
                medium_density=True,
            )
        state = np.concatenate([eef_pos, eef_close_amount, obj_pos, obj_quat, eef_to_obj_pos, target_pos, obj_to_target_pos, [check_grasp]])
        
    return state

def state2agentenv(state, env_name):
    if env_name in ALL_ENV_NAMES:
        # eef_pos 0:3, eef_close_amount 3:4, obj_pos 4:7, obj_quat 7:11, eef_to_obj_pos 11:14, target_pos 14:17, obj_to_target_pos 17:20
        agent_state = np.concatenate([state[0:4], state[11:14]])
        env_state = np.concatenate([state[4:11], state[14:17], state[17:21]])
    
    return agent_state, env_state

def cosine_sim(A, B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def filter_transition(reverse_obs, reverse_next_obs, predicted_reverse_next_obs, env_name, thresholds, state_max, state_min, filter_type):
    # reverse_obs: torch.FloatTensor
    # reverse_next_obs: torch.FloatTensor
    # predicted_reverse_next_obs: torch.FloatTensor
    if filter_type == 'None':
        reversible_indicies = torch.ones(len(reverse_obs)).bool().to(reverse_obs.device)

    if filter_type == 'state_max_diff':
        if env_name in ALL_ENV_NAMES:
            # eef_pos 0:3, eef_close_amount 3:4, obj_pos 4:7, obj_quat 7:11, eef_to_obj_pos 11:14, target_pos 14:17, obj_to_target_pos 17:20
            diff_threshold = thresholds[0]
            obj_pos, next_obj_pos = reverse_obs[:, 4:7], reverse_next_obs[:, 4:7]
            obj_true_diff = next_obj_pos - obj_pos
            obj_diff_zero_flag = torch.linalg.norm(obj_true_diff, dim=1) == 0

            state_max = torch.FloatTensor(state_max).to(reverse_obs.device)
            state_min = torch.FloatTensor(state_min).to(reverse_obs.device)
            diff = torch.abs(reverse_next_obs - predicted_reverse_next_obs) / (state_max - state_min)
            # diff = torch.cat((diff[:, 4:7]), dim=1)
            diff = diff[:, 4:7]
            diff_max, _ = torch.max(diff, dim=1)
            diff_max_flag = diff_max < diff_threshold
            
            reversible_indicies = diff_max_flag.float() + obj_diff_zero_flag.float() >= 1
    
    return reversible_indicies

def state_reward(state, env_name, reward_shaping = False):
    if env_name in ['assembly']:
        obj_pos = state[4:7]
        wrench_center = obj_pos + np.array([-0.13, 0.0, 0.0])
        target_pos = state[14:17]
        pos_error = target_pos - wrench_center
        radius = np.linalg.norm(pos_error[:2])
        aligned = radius < 0.02
        hooked = pos_error[2] > 0.0
        success = bool(aligned and hooked)

        reward = float(success)

    if env_name in ['basketball']:
        obj_to_target_pos = state[17:20]
        check_grasp = state[20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 and obj_to_target_pos[2] > 0 and check_grasp > 0.5 else 0

    if env_name in ['bin-picking']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 else 0

    if env_name in ['box-close']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 else 0
    
    if env_name in ['button-press', 'button-press-wall', 'button-press-topdown', 'button-press-topdown-wall', 'coffee-button']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.03 else 0
    
    if env_name in ['dial-turn']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 else 0

    if env_name in ['disassemble']:
        obj_pos = state[4:7]
        wrench_center = obj_pos + np.array([-0.13, 0.0, 0.0])
        target_pos = state[14:17]
        pos_error = target_pos - wrench_center
        radius = np.linalg.norm(pos_error[:2])
        aligned = radius < 0.02
        unhooked = pos_error[2] < 0.0
        success = bool(aligned and unhooked)
        reward = float(success)

    if env_name in ['coffee-pull', 'coffee-push']:
        obj_to_target_pos = state[17:20]
        check_grasp = state[20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.07 and check_grasp >= 0.5 else 0

    if env_name in ['door-open', 'door-close']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 else 0

    if env_name in ['door-lock', 'door-unlock']:
        obj_to_target_pos_z = state[19]
        reward = 1 if np.abs(obj_to_target_pos_z) < 0.03 else 0

    if env_name in ['drawer-open', 'drawer-close']:
        obj_to_target_pos_xy = state[17:19]
        reward = 1 if np.linalg.norm(obj_to_target_pos_xy) < 0.03 else 0

    if env_name in ['faucet-open', 'faucet-close']:
        obj_to_target_pos_xy = state[17:19]
        reward = 1 if np.linalg.norm(obj_to_target_pos_xy) < 0.05 else 0

    if env_name in ['hammer']:
        obj_to_target_pos = state[17:20]
        check_grasp = state[20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.1 and check_grasp >= 0.5 else 0

    if env_name in ['hand-insert']:
        obj_to_target_pos = state[17:20]
        check_grasp = state[20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.1 and check_grasp >= 0.5 else 0

    if env_name in ['handle-press-side', 'handle-pull-side', 'handle-press', 'handle-pull']:
        obj_to_target_pos_z = state[19]
        reward = 1 if np.abs(obj_to_target_pos_z) < 0.05 else 0

    if env_name in ['lever-pull']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.1 else 0

    if env_name in ['peg-insert-side', 'peg-unplug-side']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.1 else 0

    if env_name in ['pick-out-of-hole']:
        obj_to_target_pos = state[17:20]
        check_grasp = state[20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 and check_grasp >= 0.5 else 0
    
    if env_name in ['pick-place', 'pick-place-wall']:
        obj_to_target_pos = state[17:20]
        check_grasp = state[20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 and check_grasp >= 0.5 else 0

    if env_name in ['push', 'push-back', 'push-wall']:
        obj_to_target_pos = state[17:19]
        check_grasp = state[20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.1 and check_grasp >= 0.5 else 0

    if env_name in ['plate-slide', 'plate-slide-back', 'plate-slide-side', 'plate-slide-back-side']:
        obj_to_target_pos_xy = state[17:19]
        reward = 1 if np.linalg.norm(obj_to_target_pos_xy) < 0.05 else 0

    if env_name in ['reach', 'reach-wall']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 else 0

    if env_name in ['shelf-place']:
        obj_to_target_pos = state[17:20]
        check_grasp = state[20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 and check_grasp >= 0.5 else 0

    if env_name in ['soccer']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 else 0

    if env_name in ['stick-push', 'stick-pull']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.15 else 0

    if env_name in ['sweep', 'sweep-into']:
        obj_to_target_pos = state[17:20]
        reward = 1 if np.linalg.norm(obj_to_target_pos) < 0.05 else 0

    if env_name in ['window-open', 'window-close']:
        obj_to_target_pos_x = state[17]
        tolerance = 0.05 if env_name == 'window-open' else 0.025
        reward = 1 if np.abs(obj_to_target_pos_x) < tolerance else 0

    return reward

def change_target_pos(state, env_name, target_pos):
    # eef_pos 0:3, eef_close_amount 3:4, obj_pos 4:7, obj_quat 7:11, eef_to_obj_pos 11:14, target_pos 14:17, obj_to_target_pos 17:20
    if env_name in ALL_ENV_NAMES:
        tmp = state.copy()
        tmp[14:17] = target_pos
        obj_pos = tmp[4:7]
        tmp[17:20] = obj_pos - target_pos

    return tmp

def generate_transitions(states, actions, env_name, reward_shaping, change_target=True):
    transitions = []
    for i_transition in range(len(states)-1):
        agent_state = states[i_transition]
        agent_new_state = states[i_transition+1]
        action = actions[i_transition]

        if change_target:
            # change target_pos
            agent_state = change_target_pos(agent_state, env_name)
            agent_new_state = change_target_pos(agent_new_state, env_name)
        
        # compute reward
        reward = state_reward(agent_new_state, env_name, reward_shaping)
        
        done = False
        transition = [agent_state, action, reward, agent_new_state, done]
        transitions.append(transition)
    
    return transitions

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
            
def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu