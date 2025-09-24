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

def obs2state(obs, env, env_name):
    if env_name in ['Door', 'Door_Close', 'Old_Door', 'Old_Door_Close']:
        # convert the obs from the door environment to states
        # print(len(obs['robot0_eef_pos']), len(obs['robot0_gripper_qpos']), len(obs['door_pos']), len(obs['handle_pos']), len(obs['door_to_eef_pos']), len(obs['handle_to_eef_pos']), len([obs['hinge_qpos']]), len([obs['handle_qpos']])), check_grasp
        # robot0_eef_pos, robot0_gripper_qpos, door_pos, handle_pos, door_to_eef_pos, handle_to_eef_pos, hinge_qpos, handle_qpos, check_grasp
        # output: 3 6 3 3 3 3 1 1 1
        check_grasp = float(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.door.important_sites["handle"]))
        state = np.concatenate([obs['robot0_eef_pos'], obs['robot0_gripper_qpos'], obs['door_pos'], obs['handle_pos'], obs['door_to_eef_pos'], obs['handle_to_eef_pos'], [obs['hinge_qpos']], [obs['handle_qpos']], [check_grasp]])

    if env_name in ['NutAssemblyRound', 'NutDisAssemblyRound']:
        # convert the obs from the nut assembly round environment to states
        # eef_pos, nut_pos, peg_pos, goal_pos, eef_to_nut_pos, nut_to_peg_pos, nut_to_goal_pos, check_grasp
        # output: 3, 3, 3, 3, 3, 3, 3, 1
        check_grasp = float(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.nuts[1].contact_geoms[8]))
        eef_pos = obs['robot0_eef_pos']
        roundnut_pos = obs['RoundNut_pos']
        peg_pos = np.array(env.sim.data.body_xpos[env.peg2_body_id])
        goal_pos = env.goal_pos
        
        state = np.concatenate([eef_pos, roundnut_pos, peg_pos, goal_pos, eef_pos-roundnut_pos, roundnut_pos-peg_pos, roundnut_pos-goal_pos, [check_grasp]])

    if env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
        # convert the obs from the two arm peg environment to states
        # robot0_eef_pos, peg_pos, peg_quat, hole_pos, goal_pos, peg_to_goal_pos, peg_to_hole_pos, t, d, cos, check_contact_peg_hole
        # output: 3, 3, 4, 3, 3, 3, 3, 1, 1, 1, 1
        robot0_eef_pos = obs['robot0_eef_pos']
        peg_pos = env.sim.data.body_xpos[env.peg_body_id]
        peg_quat = env.sim.data.body_xquat[env.peg_body_id]
        hole_pos = env.sim.data.body_xpos[env.hole_body_id]
        goal_pos = env.goal_pos
        peg_to_goal_pos = peg_pos - goal_pos
        peg_to_hole_pos = peg_pos - hole_pos
        t, d, cos = env._compute_orientation()
        check_contact_peg_hole = float(env.check_contact(env.peg, env.hole))

        state = np.concatenate([robot0_eef_pos, peg_pos, peg_quat, hole_pos, goal_pos, peg_to_goal_pos, peg_to_hole_pos, [t], [d], [cos], [check_contact_peg_hole]])

    if env_name in ['Stack', 'UnStack']:
        # convert the obs from the block stack/unstack environment to states
        # roboo0_eef_pos, cubeA_pos, cubeB_pos, initial_cubeB_xy, goal_pos, eef_to_cubeA_pos, cubeA_to_goal_pos, cubeB_to_init_cubeB_xy, check_grasp_cubeA, cubeA_touching_cubeB
        # output: 3, 3, 3, 2, 3, 3, 3, 2, 1, 1
        robot0_eef_pos = obs['robot0_eef_pos']
        cubeA_pos = obs['cubeA_pos']
        cubeB_pos = obs['cubeB_pos']
        initial_cubeB_xy = env.initial_cubeB_xy
        goal_pos = env.goal_pos
        eef_to_cubeA_pos = robot0_eef_pos - cubeA_pos
        cubeA_to_goal_pos = cubeA_pos - goal_pos
        cubeB_to_init_cubeB_xy = cubeB_pos[0:2] = initial_cubeB_xy
        check_grasp_cubeA = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA)
        cubeA_touching_cubeB = env.check_contact(env.cubeA, env.cubeB)
        state = np.concatenate([robot0_eef_pos, cubeA_pos, cubeB_pos, initial_cubeB_xy, goal_pos, eef_to_cubeA_pos, cubeA_to_goal_pos, cubeB_to_init_cubeB_xy, [check_grasp_cubeA], [cubeA_touching_cubeB]])

    if env_name in ['PickPlaceBread', 'PickPlaceRightBread']:
        # convert the obs from the pick_place_bread environment to states
        # output: 3 6 3 4 3 3 3
        goal_pos = env.target_bin_placements[0]
        state = np.concatenate([obs['robot0_eef_pos'], obs['robot0_gripper_qpos'], obs['Bread_pos'], obs['Bread_quat'], obs['Bread_to_robot0_eef_pos'], goal_pos, obs['Bread_pos']-goal_pos])

    if env_name in ['PickPlaceCan', 'PickPlaceRightCan']:
        # convert the obs from the pick_place_can environment to states
        # output: 3 6 3 4 3 3 3
        goal_pos = env.target_bin_placements[1]
        state = np.concatenate([obs['robot0_eef_pos'], obs['robot0_gripper_qpos'], obs['Can_pos'], obs['Can_quat'], obs['Can_to_robot0_eef_pos'], goal_pos, obs['Can_pos']-goal_pos])

    if env_name in ['PickPlaceCereal', 'PickPlaceRightCereal']:
        # convert the obs from the pick_place_cereal environment to states
        # output: 3 6 3 4 3 3 3
        goal_pos = env.target_bin_placements[2]
        state = np.concatenate([obs['robot0_eef_pos'], obs['robot0_gripper_qpos'], obs['Cereal_pos'], obs['Cereal_quat'], obs['Cereal_to_robot0_eef_pos'], goal_pos, obs['Cereal_pos']-goal_pos])

    if env_name in ['PickPlaceMilk', 'PickPlaceRightMilk']:
        # convert the obs from the pick_place_milk environment to states
        # output: 3 6 3 4 3 3 3
        goal_pos = env.target_bin_placements[3]
        state = np.concatenate([obs['robot0_eef_pos'], obs['robot0_gripper_qpos'], obs['Milk_pos'], obs['Milk_quat'], obs['Milk_to_robot0_eef_pos'], goal_pos, obs['Milk_pos']-goal_pos])

    return state

def state2agentenv(state, env_name):
    if env_name in ['Door', 'Door_Close', 'Old_Door', 'Old_Door_Close']:
        # output: 3 6 3 3 3 3 1 1 1
        # robot0_eef_pos, robot0_gripper_qpos, door_pos, handle_pos, door_to_eef_pos, handle_to_eef_pos, hinge_qpos, handle_qpos, check_grasp
        # agent_state: robot0_eef_pos, robot0_gripper_qpos, door_to_eef_pos, handle_to_eef_pos, check_grasp
        # env_state: door_pos, handle_pos, hinge_qpos, handle_qpos
        agent_state = np.concatenate([state[0:9], state[15:21], state[23:24]])
        env_state = np.concatenate([state[9:15], state[21:23]])

    if env_name in ['NutAssemblyRound', 'NutDisAssemblyRound']:
        # output: 3, 3, 3, 3, 3, 3, 3, 1
        # eef_pos, nut_pos, peg_pos, goal_pos, eef_to_nut_pos, nut_to_peg_pos, nut_to_goal_pos, check_grasp
        # agent_state: eef_pos, eef_to_nut_pos, check_grasp 
        # env_state: nut_pos, peg_pos, goal_pos, nut_to_peg_pos, nut_to_goal_pos
        agent_state = np.concatenate([state[0:3], state[12:15], state[21:22]])
        env_state = np.concatenate([state[3:6], state[6:9], state[9:12], state[15:18], state[18:21]])

    if env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
        # output: 3, 3, 4, 3, 3, 3, 3, 1, 1, 1, 1
        # robot0_eef_pos, peg_pos, peg_quat, hole_pos, goal_pos, peg_to_goal_pos, peg_to_hole_pos, t, d, cos, check_contact_peg_hole
        # agent_state: robot0_eef_pos
        # env_state: peg_pos, peg_quat, hole_pos, goal_pos, peg_to_goal_pos, peg_to_hole_pos, t, d, cos, check_contact_peg_hole
        agent_state = state[0:3]
        env_state = state[3:26]

    if env_name in ['Stack', 'UnStack']:
        # output: 3, 3, 3, 2, 3, 3, 3, 2, 1, 1
        # roboo0_eef_pos, cubeA_pos, cubeB_pos, initial_cubeB_xy, goal_pos, eef_to_cubeA_pos, cubeA_to_goal_pos, cubeB_to_init_cubeB_xy, check_grasp_cubeA, cubeA_touching_cubeB
        # agent_state: roboo0_eef_pos, eef_to_cubeA_pos, check_grasp_cubeA
        # env_state: cubeA_pos, cubeB_pos, initial_cubeB_xy, goal_pos, cubeA_to_goal_pos, cubeB_to_init_cubeB_xy, cubeA_touching_cubeB
        agent_state = np.concatenate([state[0:3], state[14:17], state[22:23]])
        env_state = np.concatenate([state[3:14], state[17:22], state[22:24]])        

    return agent_state, env_state

def cosine_sim(A, B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def filter_transition(reverse_obs, reverse_next_obs, predicted_reverse_next_obs, env_name, thresholds, state_max, state_min, filter_type):
    # reverse_obs: torch.FloatTensor
    # reverse_next_obs: torch.FloatTensor
    # predicted_reverse_next_obs: torch.FloatTensor
    if filter_type == 'None':
        reversible_indicies = torch.ones(len(reverse_obs)).bool().to(reverse_obs.device)
    
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
    if env_name in ['Door', 'Door_Close', 'Old_Door', 'Old_Door_Close']:
        # output: 3 6 3 3 3 3 1 1 1
        # robot0_eef_pos, robot0_gripper_qpos, door_pos, handle_pos, door_to_eef_pos, handle_to_eef_pos, hinge_qpos, handle_qpos, check_grasp
        if filter_type == 'object_state_norm_direction':
            norm_threshold, cosine_sim_threshold = thresholds
            
            door_pos, next_door_pos, predicted_next_door_pos = reverse_obs[:, 9:12], reverse_next_obs[:, 9:12], predicted_reverse_next_obs[:, 9:12]
            door_true_diff, door_predict_diff = next_door_pos - door_pos, predicted_next_door_pos - door_pos
            door_cossim, door_norm_ratio = cosine_sim(door_true_diff, door_predict_diff), torch.linalg.norm(door_true_diff, dim=1) / torch.linalg.norm(door_predict_diff, dim=1)
            
            door_cossim_flag = door_cossim > cosine_sim_threshold
            door_diff_norm_flag1 = 1 / norm_threshold < door_norm_ratio
            door_diff_norm_flag2 = door_norm_ratio < norm_threshold
            door_diff_zero_flag = torch.linalg.norm(door_true_diff, dim=1) == 0
            
            # (cossim > cossim_threshold and 1 / norm_threshold < norm < norm_threshold) or true_diff == 0
            reversible_indicies = (door_cossim_flag.float() + door_diff_norm_flag1.float() + door_diff_norm_flag2.float() == 3).float() + door_diff_zero_flag.float() >= 1

        if filter_type == 'state_norm_direction':
            norm_threshold, cosine_sim_threshold = thresholds
            
            door_pos, next_door_pos, predicted_next_door_pos = reverse_obs[:, 9:12], reverse_next_obs[:, 9:12], predicted_reverse_next_obs[:, 9:12]
            door_true_diff, door_predict_diff = next_door_pos - door_pos, predicted_next_door_pos - door_pos
            door_cossim, door_norm_ratio = cosine_sim(door_true_diff, door_predict_diff), torch.linalg.norm(door_true_diff, dim=1) / torch.linalg.norm(door_predict_diff, dim=1)
            
            door_cossim_flag = door_cossim > cosine_sim_threshold
            door_diff_norm_flag1 = 1 / norm_threshold < door_norm_ratio
            door_diff_norm_flag2 = door_norm_ratio < norm_threshold
            door_diff_zero_flag = torch.linalg.norm(door_true_diff, dim=1) == 0
            
            eef_pos, next_eef_pos, predicted_next_eef_pos = reverse_obs[:, 0:3], reverse_next_obs[:, 0:3], predicted_reverse_next_obs[:, 0:3]
            eef_true_diff, eef_predict_diff = next_eef_pos - eef_pos, predicted_next_eef_pos - eef_pos
            eef_cossim, eef_norm_ratio = cosine_sim(eef_true_diff, eef_predict_diff), torch.linalg.norm(eef_true_diff, dim=1) / torch.linalg.norm(eef_predict_diff, dim=1)
            eef_cossim_flag, eef_diff_norm_flag1, eef_diff_norm_flag2 = eef_cossim > cosine_sim_threshold, 1 / norm_threshold < eef_norm_ratio, eef_norm_ratio < norm_threshold
            
            reversible_indicies = (door_cossim_flag.float() + door_diff_norm_flag1.float() + door_diff_norm_flag2.float() + eef_cossim_flag.float() + eef_diff_norm_flag1.float() + eef_diff_norm_flag2.float() == 6).float() + door_diff_zero_flag.float() >= 1

        if filter_type == 'state_max_diff':
            diff_threshold = thresholds[0]
            
            door_pos, next_door_pos = reverse_obs[:, 9:12], reverse_next_obs[:, 9:12]
            door_true_diff = next_door_pos - door_pos
            door_diff_zero_flag = torch.linalg.norm(door_true_diff, dim=1) == 0
            # print('door_diff_zero_flag', door_diff_zero_flag)

            state_max = torch.FloatTensor(state_max).to(reverse_obs.device)
            state_min = torch.FloatTensor(state_min).to(reverse_obs.device)
            diff = torch.abs(reverse_next_obs - predicted_reverse_next_obs) / (state_max - state_min)
            diff = torch.cat((diff[:, 0:3], diff[:, 9:11], diff[:, 12:15], diff[:, 21:23]), dim=1)
            diff_max, _ = torch.max(diff, dim=1)
            diff_max_flag = diff_max < diff_threshold
            # print('diff_max_flag', diff_max_flag)

            reversible_indicies = diff_max_flag.float() + door_diff_zero_flag.float() >= 1

    if env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
        # output: 3, 3, 4, 3, 3, 3, 3, 1, 1, 1, 1
        # robot0_eef_pos, peg_pos, peg_quat, hole_pos, goal_pos, peg_to_goal_pos, peg_to_hole_pos, t, d, cos, check_contact_peg_hole
        if filter_type == 'state_max_diff':
            diff_threshold = thresholds[0]
            
            hole_pos, next_hole_pos = reverse_obs[:, 10:13], reverse_next_obs[:, 10:13]
            hole_true_diff = next_hole_pos - hole_pos
            hole_diff_zero_flag = torch.linalg.norm(hole_true_diff, dim=1) == 0

            state_max = torch.FloatTensor(state_max).to(reverse_obs.device)
            state_min = torch.FloatTensor(state_min).to(reverse_obs.device)
            diff = torch.abs(reverse_next_obs - predicted_reverse_next_obs) / (state_max - state_min)
            diff = torch.cat((diff[:, 0:6], diff[:, 10:13]), dim=1)
            diff_max, _ = torch.max(diff, dim=1)
            diff_max_flag = diff_max < diff_threshold
            # print('diff_max_flag', diff_max_flag)

            reversible_indicies = diff_max_flag.float() + hole_diff_zero_flag.float() >= 1
    
    if env_name in ['NutAssemblyRound', 'NutDisAssemblyRound']:
        # output: 3, 3, 3, 3, 3, 3, 3, 1
        # eef_pos, nut_pos, peg_pos, goal_pos, eef_to_nut_pos, nut_to_peg_pos, nut_to_goal_pos, check_grasp
        if filter_type == 'state_max_diff':
            diff_threshold = thresholds[0]

            nut_pos, next_nut_pos = reverse_obs[:, 3:6], reverse_next_obs[:, 3:6]
            nut_true_diff = next_nut_pos - nut_pos
            nut_diff_zero_flag = torch.linalg.norm(nut_true_diff, dim=1) == 0
            
            state_max = torch.FloatTensor(state_max).to(reverse_obs.device)
            state_min = torch.FloatTensor(state_min).to(reverse_obs.device)
            diff = torch.abs(reverse_next_obs - predicted_reverse_next_obs) / (state_max - state_min)
            diff = diff[:, 3:6]
            diff_max, _ = torch.max(diff, dim=1)
            diff_max_flag = diff_max < diff_threshold
    
            reversible_indicies = diff_max_flag.float() + nut_diff_zero_flag.float() >= 1

    if env_name in ['Stack', 'UnStack']:
        # output: 3, 3, 3, 2, 3, 3, 3, 2, 1, 1
        # roboo0_eef_pos, cubeA_pos, cubeB_pos, initial_cubeB_xy, goal_pos, eef_to_cubeA_pos, cubeA_to_goal_pos, cubeB_to_init_cubeB_xy, check_grasp_cubeA, cubeA_touching_cubeB
        if filter_type == 'state_max_diff':
            diff_threshold = thresholds[0]

            cubeA_pos, next_cubeA_pos = reverse_obs[:, 3:6], reverse_next_obs[:, 3:6]
            cubeA_true_diff = next_cubeA_pos - cubeA_pos
            cubeA_diff_zero_flag = torch.linalg.norm(cubeA_true_diff, dim=1) == 0
            
            state_max = torch.FloatTensor(state_max).to(reverse_obs.device)
            state_min = torch.FloatTensor(state_min).to(reverse_obs.device)
            diff = torch.abs(reverse_next_obs - predicted_reverse_next_obs) / (state_max - state_min)
            diff = diff[:, 3:6]
            diff_max, _ = torch.max(diff, dim=1)
            diff_max_flag = diff_max < diff_threshold
    
            reversible_indicies = diff_max_flag.float() + cubeA_diff_zero_flag.float() >= 1

    return reversible_indicies

def change_goal_pos(state, env_name, goal_pos=[]):
    if env_name in ['Door', 'Door_Close', 'Old_Door', 'Old_Door_Close']:
        return state

    if env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
        tmp = state.copy()
        if len(goal_pos) == 0: 
            if env_name == 'TwoArmPegInHole': goal_pos = np.array([-0.06, 0.21, 0.93])
            if env_name == 'TwoArmPegRemoval': goal_pos = np.array([-0.2, -0.25, 0.95])
        tmp[13:16] = goal_pos
        peg_pos = tmp[3:6]
        peg_to_goal_pos = peg_pos - goal_pos
        tmp[16:19] = peg_to_goal_pos 
        return tmp

    if env_name in ['NutAssemblyRound', 'NutDisAssemblyRound']:
        # eef_pos, nut_pos, peg_pos, goal_pos, eef_to_nut_pos, nut_to_peg_pos, nut_to_goal_pos, check_grasp
        # output: 3, 3, 3, 3, 3, 3, 3, 1
        
        tmp = state.copy()
        tmp[9:12] = goal_pos
        nut_pos = tmp[3:6]
        nut_to_goal_pos = nut_pos - goal_pos
        tmp[18:21] = nut_to_goal_pos

        return tmp

    if env_name in ['Stack', 'UnStack']:
        # roboo0_eef_pos, cubeA_pos, cubeB_pos, initial_cubeB_xy, goal_pos, eef_to_cubeA_pos, cubeA_to_goal_pos, cubeB_to_init_cubeB_xy, check_grasp_cubeA, cubeA_touching_cubeB
        # output: 3, 3, 3, 2, 3, 3, 3, 2, 1, 1
        init_cubeB_xy = goal_pos[0:2]
        goal_pos = goal_pos[2:5]
        
        tmp = state.copy()
        tmp[9:11] = init_cubeB_xy
        tmp[11:14] = goal_pos
        cubeA_pos = tmp[3:6]
        cubeB_pos = tmp[6:9]
        cubeA_to_goal_pos = cubeA_pos - goal_pos
        cubeB_to_init_cubeB_xy = cubeB_pos[0:2] - init_cubeB_xy
        tmp[17:20] = cubeA_to_goal_pos
        tmp[20:22] = cubeB_to_init_cubeB_xy

        return tmp

def reverse_action(actions, env_name):
    if env_name == 'Door':
        reversed_actions = actions[::-1]
        reversed_actions = reversed_actions * np.array([-1, -1, -1, 1])
    
    return reversed_actions

def state_reward(state, env_name, reward_shaping, info):
    agent_state = state

    if env_name == 'Door':
        reward = 0
        hinge_qpos = agent_state[21]
        check_grasp = agent_state[23]
        
        # check success
        if check_grasp and hinge_qpos < -0.25:
            return 1

        if reward_shaping:
            # Add reaching component
            gripper_to_handle = agent_state[18:21]
            dist = np.linalg.norm(gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            
            # Add grasping handle
            if check_grasp:
                reward += 0.25
            
            if check_grasp:
                # Add rotating component if we're using a locked door
                handle_qpos = agent_state[22]
                reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)
                
                # Add rotating the door
                reward += 0.25 * np.clip(-hinge_qpos / 0.3, 0, 1)
    
    if env_name == 'Door_Close':
        reward = 0
        hinge_qpos = agent_state[21]
        check_grasp = agent_state[23]
        
        # sparse completion reward
        if check_grasp and hinge_qpos > -0.05:
            return 1.0

        if reward_shaping:
            # Add reaching component
            gripper_to_handle = agent_state[18:21]
            dist = np.linalg.norm(gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            
            # Add grasping handle
            if check_grasp:
                reward += 0.25

            if check_grasp:       
                # Add rotating component if we're using a locked door
                handle_qpos = agent_state[22]
                reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)
                
                # Add rotating the door
                reward += 0.25 * np.clip((0.3+hinge_qpos) / 0.3, 0, 1)

    if env_name == 'Old_Door':
        reward = 0
        hinge_qpos = agent_state[21]
        handle_qpos = agent_state[22]
        check_grasp = agent_state[23]
        
        # check success
        # if check_grasp and hinge_qpos > 0.3:
        if hinge_qpos > 0.3 and np.abs(handle_qpos) < 0.05:
            return 1

        if reward_shaping:
            # Add reaching component
            gripper_to_handle = agent_state[18:21]
            dist = np.linalg.norm(gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            
            # Add grasping handle
            if check_grasp:
                reward += 0.25
            
            # if check_grasp:
            # Add rotating component if we're using a locked door
            handle_qpos = agent_state[22]
            reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)
            
            # Add rotating the door
            reward += 0.25 * np.clip(hinge_qpos / 0.3, 0, 1)

    if env_name == 'Old_Door_Close':
        reward = 0
        hinge_qpos = agent_state[21]
        handle_qpos = agent_state[22]
        check_grasp = agent_state[23]
        
        # sparse completion reward
        # if check_grasp and hinge_qpos < 0.01:
        if hinge_qpos < 0.01 and np.abs(handle_qpos) < 0.05:
            return 1.0

        if reward_shaping:
            # Add reaching component
            gripper_to_handle = agent_state[18:21]
            dist = np.linalg.norm(gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            
            # Add grasping handle
            if check_grasp:
                reward += 0.25

            # if check_grasp:       
            # Add rotating component if we're using a locked door
            handle_qpos = agent_state[22]
            reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)
            
            # Add rotating the door
            reward += 0.25 * np.clip((0.3-hinge_qpos) / 0.3, 0, 1)

    if env_name == 'TwoArmPegInHole':
        # output: 3, 3, 4, 3, 3, 3, 3, 1, 1, 1, 1
        # robot0_eef_pos, peg_pos, peg_quat, hole_pos, goal_pos, peg_to_goal_pos, peg_to_hole_pos, t, d, cos, check_contact_peg_hole
        
        reward = 0.0
        peg_pos = state[3:6]
        goal_pos = state[13:16]
        dist_peg_to_goal = np.linalg.norm(peg_pos - goal_pos)

        hole_pos = state[10:13]
        hole_init_pos = info['hole_init_pos']
        dist_hole_to_init = np.linalg.norm(hole_pos - hole_init_pos)

        t, d, cos = state[22], state[23], state[24]
        check_contact_peg_hole = state[25]
        
        if d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95 and dist_hole_to_init < 0.02 and check_contact_peg_hole == 0:
            reward = 1.0

    if env_name == 'TwoArmPegRemoval':
        reward = 0

        peg_pos = state[3:6]
        goal_pos = state[13:16]
        dist_peg_to_goal = np.linalg.norm(peg_pos - goal_pos)
        
        hole_pos = state[10:13]
        hole_init_pos = info['hole_init_pos']
        dist_hole_to_init = np.linalg.norm(hole_pos - hole_init_pos)

        check_contact_peg_hole = state[25]
        
        if dist_peg_to_goal < 0.05 and dist_hole_to_init < 0.02 and check_contact_peg_hole == 0:
            reward = 1.0

    if env_name == 'NutAssemblyRound':
        reward = 0

        nut_pos = state[3:6]
        peg_pos = state[6:9]
        goal_pos = state[9:12]
        nut_fit_peg = np.linalg.norm(nut_pos[0:2] - peg_pos[0:2]) < 0.05

        dist_goal_to_peg_z = np.abs(goal_pos[2] - nut_pos[2])
        goal_peg_same_height = dist_goal_to_peg_z < 0.05

        if nut_fit_peg and goal_peg_same_height:
            reward = 1.0

    if env_name == 'NutDisAssemblyRound':
        reward = 0

        nut_pos = state[3:6]
        goal_pos = state[9:12]
        dist_nut_to_goal = np.linalg.norm(nut_pos - goal_pos)

        if dist_nut_to_goal < 0.05:
            reward = 1.0

    if env_name == 'Stack':
        # output: 3, 3, 3, 2, 3, 3, 3, 2, 1, 1
        # roboo0_eef_pos, cubeA_pos, cubeB_pos, initial_cubeB_xy, goal_pos, eef_to_cubeA_pos, cubeA_to_goal_pos, cubeB_to_init_cubeB_xy, check_grasp_cubeA, cubeA_touching_cubeB
        cubeA_pos = state[3:6]
        cubeB_pos = state[6:9]
        initial_cubeB_xy = state[9:11]
        check_grasp_cubeA = state[22]
        
        success = False
        cubeB_not_move = np.linalg.norm(cubeB_pos[0:2] - initial_cubeB_xy) < 0.05
        cubeA_on_cubeB = np.abs(cubeA_pos[0] - cubeB_pos[0]) < 0.03 and np.abs(cubeA_pos[1] - cubeB_pos[1]) < 0.03 and cubeA_pos[2] > cubeB_pos[2]
        success = check_grasp_cubeA and cubeB_not_move and cubeA_on_cubeB
    
        reward = float(success)

    if env_name == 'UnStack':
        cubeA_pos = state[3:6]
        cubeB_pos = state[6:9]
        initial_cubeB_xy = state[9:11]
        goal_pos = state[11:14]
        check_grasp_cubeA = state[22]

        success = False
        cubeB_not_move = np.linalg.norm(cubeB_pos[0:2] - initial_cubeB_xy) < 0.05
        cubeA_on_goal = np.linalg.norm(cubeA_pos - goal_pos) < 0.05
        success = check_grasp_cubeA and cubeB_not_move and cubeA_on_goal

        reward = float(success)

    return reward

def generate_transitions(states, actions, dones, env_name, reward_shaping):
    # generate transitions from one trajectory of states, actions and dones

    transitions = []
    for i_transition in range(len(states)-1):
        agent_state = states[i_transition]
        agent_new_state = states[i_transition+1]
        action = actions[i_transition]
        done = dones[i_transition]

        new_state = states[i_transition+1]
        info = {}
        if env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
            info['hole_init_pos'] = states[0][10:13]
        reward = state_reward(new_state, env_name, reward_shaping, info)
        
        transition = [agent_state, action, reward, agent_new_state, done]
        transitions.append(transition)
    
    return transitions

def get_reverse_batch_size(i_episode, num_training_episodes, batch_size):
    return int(batch_size * min(i_episode / (num_training_episodes // 2), 1))

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


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


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


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


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


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