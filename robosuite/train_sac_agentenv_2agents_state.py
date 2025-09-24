import copy
import math
import matplotlib.pyplot as plt
from mujoco_py.builder import MujocoException
import numpy as np
import os
import robosuite as suite
from robosuite.controllers import load_controller_config
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import obs2state, state2agentenv, state_reward, change_goal_pos
import argparse
import imageio
import pandas as pd
from sac_agentenv import SACAgent

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

import warnings 
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env1_name', type = str, help = 'env1_name', default = 'Old_Door')
    parser.add_argument('--env2_name', type = str, help = 'env2_name', default = 'Old_Door_Close')
    parser.add_argument('--batch_size', type = int, help = 'batch_size', default = 512)
    parser.add_argument('--hidden_units', type = int, help = 'hidden_units', default = 512)
    parser.add_argument('--seed', type = int, help = 'seed', default = 1)
    parser.add_argument('--horizon', type = int, help = 'horizon', default = 500)
    parser.add_argument('--reward_shaping', type=str, default='true')
    parser.add_argument('--n_demo', type = int, help = 'n_demo', default = 0)
    parser.add_argument('--demo_env', type=str, default = 'both') # both, env1, env2
    parser.add_argument('--use_forward_reward', type=str, default = 'false')
    parser.add_argument('--use_reversed_reward', type=str, default = 'false')
    parser.add_argument('--n_forward_demo', type = int, help = 'n_forward_demo', default = 10)
    parser.add_argument('--n_reverse_demo', type = int, help = 'n_reverse_demo', default = 10)
    parser.add_argument('--reward_model_type', type=str, default = 'reward') # reward, potential
    parser.add_argument('--potential_type', type=str, default = 'linear') # linear, triangular, geometric, original_geometric, constant
    parser.add_argument('--use_reversed_transition', type=str, default = 'false')
    parser.add_argument('--diff_threshold', type=float, default = 0.01)
    parser.add_argument('--filter_type', type=str, default = 'None') # None, state_norm_direction, object_state_norm_direction, state_max_diff
    args = vars(parser.parse_args())

    env1_name = args['env1_name']
    env2_name = args['env2_name']
    env_names = [env1_name, env2_name]
    num_envs = len(env_names)
    print('env_names', env_names)
    batch_size = args['batch_size']
    hidden_depth = 2
    hidden_dim = args['hidden_dim']
    print('batch_size {} hidden_depth {} hidden_dim {}'.format(batch_size, hidden_depth, hidden_dim))

    seed = args['seed']
    print('seed {}'.format(seed))
    utils.set_seed_everywhere(seed)
    horizon = args['horizon']
    print('horizon {}'.format(horizon))
    reward_shaping = True if args['reward_shaping'] == 'true' else False
    print('reward_shaping {}'.format(reward_shaping))
    n_demo = args['n_demo']
    print('n_demo', n_demo)
    n_forward_demo = args['n_forward_demo']
    print('n_forward_demo', n_forward_demo)
    n_reverse_demo = args['n_reverse_demo']
    print('n_reverse_demo', n_reverse_demo)
    demo_env = args['demo_env']
    print('demo_env', demo_env)
    use_forward_reward = False if args['use_forward_reward'] == 'false' else True
    print('use_forward_reward', use_forward_reward)
    use_reversed_reward = False if args['use_reversed_reward'] == 'false' else True
    print('use_reversed_reward', use_reversed_reward)
    reward_model_type = args['reward_model_type']
    print('reward_model_type {}'.format(reward_model_type))
    use_reversed_transition = False if args['use_reversed_transition'] == 'false' else True
    print('use_reversed_transition', use_reversed_transition)
    filter_type = args['filter_type']
    print('filter_type {}'.format(filter_type))
    potential_type = args['potential_type']
    print('potential_type {}'.format(potential_type))
    diff_threshold = args['diff_threshold']
    print('diff_threshold {}'.format(diff_threshold))
    save_video = True # True
    print('save_video {}'.format(save_video))
    
    algo_name = 'SAC_agentenv_2agents' + '_{}demo'.format(n_demo)
    if demo_env != 'both': algo_name += 'from{}'.format(demo_env)
    if use_reversed_reward: algo_name += '_use_reversed_reward_{}reverse_demo'.format(n_reverse_demo)
    if use_forward_reward: algo_name += '_use_forward_reward_{}demo'.format(n_forward_demo)
    if use_reversed_reward and use_forward_reward: algo_name += '_separate'
    if reward_model_type != 'reward': algo_name += '_{}_model'.format(reward_model_type)
    if reward_model_type == 'potential': algo_name += '_{}_potential'.format(potential_type)
    thresholds = []
    if use_reversed_transition: 
        if filter_type == 'None':
            algo_name += '_use_reversed_transition'
        if filter_type == 'state_norm_direction':
            thresholds = [norm_threshold, cosine_sim_threshold]
            algo_name += '_use_reversed_transition_state_norm{}_cossim{}'.format(norm_threshold, cosine_sim_threshold)
        if filter_type == 'object_state_norm_direction':
            thresholds = [norm_threshold, cosine_sim_threshold]
            algo_name += '_use_reversed_transition_object_state_norm{}_cossim{}'.format(norm_threshold, cosine_sim_threshold)
        if filter_type == 'state_max_diff':
            thresholds = [diff_threshold]
            algo_name += '_use_reversed_transition_state_max_diff{}'.format(diff_threshold)
        
    if not reward_shaping: algo_name += '_sparse'
    if reward_shaping and (not use_forward_reward) and (not use_reversed_reward) and (not use_reversed_transition): algo_name = 'SAC'
    if algo_name == 'SAC_agentenv_2agents_10demo_use_reversed_reward_10reverse_demo_use_forward_reward_10demo_separate_potential_model_linear_potential_use_reversed_transition_state_max_diff0.01_sparse': algo_name = 'single_task_tr_sac'
    print('algo_name: {} state'.format(algo_name))

    reward_model_max_value = 0.5
    success_threshold = 400
    if env1_name in ['Door', 'Door_Close', 'Old_Door', 'Old_Door_Close', 'TwoArmPegInHole', 'TwoArmPegRemoval']:
        success_threshold = 400
    if env1_name in ['NutAssemblyRound', 'NutDisAssemblyRound', 'Stack', 'UnStack']:
        success_threshold = 350

    robots = 'Kinova3'
    controller_name = 'OSC_POSITION' # in ['JOINT_VELOCITY', 'JOINT_TORQUE', 'JOINT_POSITION', 'OSC_POSITION', 'OSC_POSE', 'IK_POSE']
    if env1_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
        robots = ['Kinova3', 'Kinova3']
        controller_name = 'OSC_POSE'
    controller_configs = load_controller_config(default_controller=controller_name)
    camera_names = ['agentview', 'sideview']
    camera_heights = 256
    camera_widths = 256
    save_video_train = True #True
    print('save_video_train {}'.format(save_video_train))
    has_renderer = False # False
    has_offscreen_renderer = True #True

    # initialize env1
    env1 = suite.make(
        env_name = env1_name, reward_shaping= reward_shaping, robots = robots, controller_configs = controller_configs,
        has_renderer = has_renderer, has_offscreen_renderer = has_offscreen_renderer, use_camera_obs = save_video_train, control_freq=20, horizon=horizon,
        camera_names = camera_names, camera_heights = camera_heights, camera_widths = camera_widths
    )
    test_env1 = suite.make(
        env_name = env1_name, reward_shaping= reward_shaping, robots = robots, controller_configs = controller_configs,
        has_renderer = has_renderer, has_offscreen_renderer = has_offscreen_renderer, use_camera_obs = save_video, control_freq=20, horizon=horizon,
        camera_names = camera_names, camera_heights = camera_heights, camera_widths = camera_widths
    )
    
    # initialize env2
    env2 = suite.make(
        env_name = env2_name, reward_shaping= reward_shaping, robots = robots, controller_configs = controller_configs,
        has_renderer = has_renderer, has_offscreen_renderer = has_offscreen_renderer, use_camera_obs = save_video_train, control_freq=20, horizon=horizon,
        camera_names = camera_names, camera_heights = camera_heights, camera_widths = camera_widths
    )
    test_env2 = suite.make(
        env_name = env2_name, reward_shaping= reward_shaping, robots = robots, controller_configs = controller_configs, 
        has_renderer = has_renderer, has_offscreen_renderer = has_offscreen_renderer, use_camera_obs = save_video, control_freq=20, horizon=horizon,
        camera_names = camera_names, camera_heights = camera_heights, camera_widths = camera_widths
    )

    # Get action size and action limits
    action_size = env1.action_dim
    action_low, action_high = env1.action_spec
    if env1_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
        action_size = env1.action_dim // 2
        action_low, action_high = -np.ones(action_size), np.ones(action_size)
    action_range = [float(action_low.min()), float(action_high.max())]
    action_shape = (action_size,)
    
    obs = env1.reset()
    state = obs2state(obs, env1, env1_name)
    agent_state, env_state = state2agentenv(state, env1_name)
    state_size, agent_state_size, env_state_size = len(state), len(agent_state), len(env_state)
    print('action size {} state size {} agent state size {} env state size {}'.format(action_size, state_size, agent_state_size, env_state_size))

    _max_episode_steps = env1.horizon

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    discount, init_temperature = 0.99, 0.1
    actor_lr, critic_lr, reward_lr, potential_lr, for_lr, inv_lr, alpha_lr = 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3
    print('actor_lr {} critic_lr {} reward_lr {} potential_lr {} for_lr {} inv_lr {} alpha_lr {}'.format(actor_lr, critic_lr, reward_lr, potential_lr, for_lr, inv_lr, alpha_lr))
    actor_update_frequency, critic_tau, critic_target_update_frequency = 2, 0.01, 2
    log_std_bounds = [-10, 2]
    replay_buffer_size, demo_replay_buffer_size, reverse_replay_buffer_size = 300000, n_demo * horizon, 300000
    
    log_dir1 = 'runs/{}/{}_seed{}/'.format(algo_name, env1_name, seed)
    if not os.path.exists(log_dir1):
        os.makedirs(log_dir1)
    train_video_dir1 = os.path.join(log_dir1, 'train_videos')
    if not os.path.exists(train_video_dir1):
        os.mkdir(train_video_dir1)
    video_dir1 = os.path.join(log_dir1, 'videos')
    if not os.path.exists(video_dir1):
        os.mkdir(video_dir1)
    model_dir1 = os.path.join(log_dir1, 'models')
    if not os.path.exists(model_dir1):
        os.mkdir(model_dir1)
    replay_buffer_dir1 = os.path.join(log_dir1, 'replay_buffer')
    if not os.path.exists(replay_buffer_dir1):
        os.mkdir(replay_buffer_dir1)
    train_df1 = pd.DataFrame(columns=['episode', 'num_steps', 'score', 'success', 'estimated_score', 'critic_loss', 'actor_loss', 'forward_reward_loss', 'reverse_reward_loss', 'reversible_ratio', 'for_loss', 'inv_loss', 'alpha_loss', 'alpha', 'log_prob', 'env_buffer', 'demo_buffer', 'success_buffer', 'reverse_buffer', 'reversal_buffer', 'train_time', 'episode_time', 'total_time'])
    train_df_path1 = os.path.join(log_dir1, 'train.csv')
    eval_df1 = pd.DataFrame(columns=['episode', 'i_test', 'num_steps', 'eval_score', 'estimated_eval_score', 'eval_success', 'episode_time', 'total_time'])
    eval_df_path1 = os.path.join(log_dir1, 'eval.csv')

    log_dir2 = 'runs/{}/{}_seed{}/'.format(algo_name, env2_name, seed)
    if not os.path.exists(log_dir2):
        os.makedirs(log_dir2)
    train_video_dir2 = os.path.join(log_dir2, 'train_videos')
    if not os.path.exists(train_video_dir2):
        os.mkdir(train_video_dir2)
    video_dir2 = os.path.join(log_dir2, 'videos')
    if not os.path.exists(video_dir2):
        os.mkdir(video_dir2)
    model_dir2 = os.path.join(log_dir2, 'models')
    if not os.path.exists(model_dir2):
        os.mkdir(model_dir2)
    replay_buffer_dir2 = os.path.join(log_dir2, 'replay_buffer')
    if not os.path.exists(replay_buffer_dir2):
        os.mkdir(replay_buffer_dir2)
    train_df2 = pd.DataFrame(columns=['episode', 'num_steps', 'score', 'success', 'estimated_score', 'critic_loss', 'actor_loss', 'forward_reward_loss', 'reverse_reward_loss', 'reversible_ratio', 'for_loss', 'inv_loss', 'alpha_loss', 'alpha', 'log_prob', 'env_buffer', 'demo_buffer', 'success_buffer', 'reverse_buffer', 'reversal_buffer', 'train_time', 'episode_time', 'total_time'])
    train_df_path2 = os.path.join(log_dir2, 'train.csv')
    eval_df2 = pd.DataFrame(columns=['episode', 'i_test', 'num_steps', 'eval_score', 'estimated_eval_score', 'eval_success', 'episode_time', 'total_time'])
    eval_df_path2 = os.path.join(log_dir2, 'eval.csv')
    
    video_dirs = [video_dir1, video_dir2]
    train_video_dirs = [train_video_dir1, train_video_dir2]
    model_dirs = [model_dir1, model_dir2]
    replay_buffer_dirs = [replay_buffer_dir1, replay_buffer_dir2]
    train_df_paths = [train_df_path1, train_df_path2]
    eval_df_paths = [eval_df_path1, eval_df_path2]

    agent1 = SACAgent(env1_name, state_size, agent_state_size, env_state_size, action_size, action_range, device,
                 discount, init_temperature, actor_lr, critic_lr, reward_lr, potential_lr, for_lr, inv_lr, alpha_lr, actor_update_frequency, 
                 critic_tau, critic_target_update_frequency, batch_size, 
                 hidden_dim, hidden_depth, log_std_bounds, 
                 demo_replay_buffer_size, reverse_replay_buffer_size, replay_buffer_size)
    agent2 = SACAgent(env2_name, state_size, agent_state_size, env_state_size, action_size, action_range, device,
                 discount, init_temperature, actor_lr, critic_lr, reward_lr, potential_lr, for_lr, inv_lr, alpha_lr, actor_update_frequency, 
                 critic_tau, critic_target_update_frequency, batch_size, 
                 hidden_dim, hidden_depth, log_std_bounds, 
                 demo_replay_buffer_size, reverse_replay_buffer_size, replay_buffer_size)

    for i_env in range(num_envs):
        if (demo_env == 'both' or (demo_env == 'env1' and i_env==0) or (demo_env=='env2' and i_env==1)):
            agent = agent1 if i_env == 0 else agent2
            reverse_agent = agent1 if i_env == 1 else agent2
            env_name = env_names[i_env]
            reverse_env_name = env_names[(i_env+1)%num_envs]
            if reward_shaping:
                transition_path = 'generate/{}_transitions_{}trajectory.npy'.format(env_name, n_demo)
            else:
                transition_path = 'generate/{}_transitions_{}trajectory_sparse.npy'.format(env_name, n_demo)
            if os.path.exists(transition_path):
                transitions = np.load(transition_path, allow_pickle=True)

                for transition in transitions:
                    state, action, reward, next_state, done = transition
                    agent_state, env_state = state2agentenv(state, env_name)
                    next_agent_state, next_env_state = state2agentenv(next_state, env_name)
                    agent.remember(state, agent_state, env_state, action, reward, next_state, next_agent_state, next_env_state, done, type='demo')
                    
                print('env {}: load {} transitions from {} to demo replay buffer'.format(env_name, len(transitions), transition_path))
                print('demo reward density {:.4f}'.format(np.mean(agent.demo_replay_buffer.reward_memory[:agent.demo_replay_buffer.mem_cntr])))
            
                if use_reversed_transition:
                    for i_demo in range(n_demo):
                        state_this_episode = []
                        act_this_episode = []
                        next_state_this_episode = []
                        done_no_max_this_episode = []
                        for i_transition in range(i_demo * horizon, (i_demo + 1) * horizon):
                            state, action, reward, next_state, done = transitions[i_transition]
                            state_this_episode.append(state)
                            act_this_episode.append(action)
                            next_state_this_episode.append(next_state)
                            done_no_max_this_episode.append(done)

                        # reverse this trajectory
                        reverse_states = next_state_this_episode.copy()
                        reverse_next_states = state_this_episode.copy()                
                        
                        # The trajectory is reversed, so the goal position should be changed accordingly
                        info = {}
                        if reverse_env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
                            if reverse_env_name == 'TwoArmPegInHole': 
                                # the reverse agent is for two arm peg in hole
                                # goal pos is set as the initial hole pos of the reversed trajectory
                                goal_pos = next_state_this_episode[-1][10:13]
                            if reverse_env_name == 'TwoArmPegRemoval':
                                # the reverse agent is for two arm peg removal
                                # goal pos is set as the last peg pos of the reversed trajectory 
                                goal_pos = state_this_episode[0][3:6]
                            # hole init pos is set as the initial hole pos of the reversed trajectory
                            info['hole_init_pos'] = next_state_this_episode[-1][10:13]
                        
                        if reverse_env_name in ['NutAssemblyRound', 'NutDisAssemblyRound']:
                            if reverse_env_name == 'NutAssemblyRound':
                                # the reverse agent is for nut assembly
                                goal_pos = np.array([0.1, -0.1, 0.85])
                            if reverse_env_name == 'NutDisAssemblyRound':
                                # the reverse agent is for nut disassembly
                                goal_pos = np.array([-0.15, -0.15, 1.0])

                        if reverse_env_name in ['Stack', 'UnStack']:
                            if reverse_env_name == 'Stack':
                                # the reverse agent is for block stack
                                # goal pos is set as the initial cubeB pos of the reversed trajectory
                                cubeB_x, cubeB_y = next_state_this_episode[-1][6], next_state_this_episode[-1][7]
                                goal_pos = np.array([cubeB_x, cubeB_y, 0.85])
                            if reverse_env_name == 'UnStack':
                                # the reverse agent is for block unstack
                                # goal pos is set as the last cubeA pos of the reversed trajectory
                                goal_pos = state_this_episode[0][3:6]
                            init_cubeB_xy = next_state_this_episode[-1][6:8]
                            goal_pos = np.concatenate([init_cubeB_xy, goal_pos])

                        if reverse_env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval', 'NutAssemblyRound', 'NutDisAssemblyRound', 'Stack', 'UnStack']:
                            reverse_states = [change_goal_pos(s, reverse_env_name, goal_pos) for s in reverse_states]
                            reverse_next_states = [change_goal_pos(s, reverse_env_name, goal_pos) for s in reverse_next_states]
                                    
                        reverse_rewards = [state_reward(s, reverse_env_name, reward_shaping, info) for s in reverse_next_states]
                        # add this trajectory to reverse agent replay buffer
                        for i_sample in range(len(reverse_states)):
                            reverse_state, act, reverse_reward, reverse_next_state, done_no_max = reverse_states[i_sample], act_this_episode[i_sample], reverse_rewards[i_sample], reverse_next_states[i_sample], done_no_max_this_episode[i_sample]

                            reverse_agent_state, reverse_env_state = state2agentenv(reverse_state, reverse_env_name)
                            reverse_next_agent_state, reverse_next_env_state = state2agentenv(reverse_next_state, reverse_env_name)

                            reverse_agent.remember(reverse_state, reverse_agent_state, reverse_env_state, act, reverse_reward, reverse_next_state, reverse_next_agent_state, reverse_next_env_state, done_no_max, type='reversal')
  
                    print('reverse env {}: load {} transitions from {} to reversal replay buffer'.format(reverse_env_name, len(transitions), transition_path))
                    print('reversal reward density {:.4f}'.format(np.mean(reverse_agent.reversal_replay_buffer.reward_memory[:reverse_agent.reversal_replay_buffer.mem_cntr])))

            else:
                print('{} does not exist.'.format(transition_path))

    # get state_max and state_min
    state_max = []
    state_min = []
    if use_reversed_transition:
        for i_env in range(num_envs):
            env_name = env_names[i_env]
            transition_path = 'generate/{}_transitions_50trajectory_sparse.npy'.format(env_name)
            if os.path.exists(transition_path):
                transitions = np.load(transition_path, allow_pickle=True)
                state = transitions[:, 0]
                state_size = len(state[0])
                state = np.concatenate(state, axis=0).reshape(-1, state_size)
                state_max.append(np.max(state, axis=0)) 
                state_min.append(np.min(state, axis=0))
                
        print('state_max {}'.format(state_max))
        print('state_min {}'.format(state_min))
    else:
        state_max = [0, 1]
        state_min = [0, 1]

    if use_forward_reward:
        for i_env in range(num_envs):
            reward_trajectory_lengths = []
            total_env_states = []
            total_rewards = []
            total_next_env_states = []
            
            agent = agent1 if i_env == 0 else agent2
            env_name = env_names[i_env]
            transition_path = 'generate/{}_transitions_{}trajectory_sparse.npy'.format(env_name, n_forward_demo)
            if os.path.exists(transition_path):
                transitions = np.load(transition_path, allow_pickle=True)
                for i_episode in range(n_forward_demo):
                    states = []
                    env_states = []
                    rewards = []
                    next_states = []
                    next_env_states = []
                    for i, transition in enumerate(transitions[i_episode*horizon:(i_episode+1)*horizon]):
                        state, action, reward, next_state, done = transition
                        agent_state, env_state = state2agentenv(state, env_name)
                        next_agent_state, next_env_state = state2agentenv(next_state, env_name)
                        
                        states.append(state)
                        env_states.append(env_state)
                        rewards.append(reward)
                        next_states.append(next_state)
                        next_env_states.append(next_env_state)
                    if 1 in rewards and np.sum(rewards) > success_threshold:
                        first_reward = rewards.index(1)
                        
                        # env_state_diff = [np.linalg.norm(next_env_states[i]-env_states[i]) for i in range(first_reward)]
                        # first_env_change = list(np.array(env_state_diff) > 1e-3).index(True)
                        # start_index = max(first_env_change - 5, 0)

                        start_index = 0
                        end_index = first_reward
                        reward_trajectory_lengths.append(end_index - start_index)
                        print(start_index, end_index)

                        for i_step in range(start_index, end_index+1):
                            state = states[i_step]
                            env_state = env_states[i_step]
                            next_state = next_states[i_step]
                            next_env_state = next_env_states[i_step]
                            total_env_states.append(env_state)
                            total_next_env_states.append(next_env_state)
                            if reward_model_type == 'reward':
                                total_rewards.append((1 / (end_index+1-start_index) * (i_step + 1)) * reward_model_max_value)
                            if reward_model_type == 'potential':
                                len_trajectory = end_index + 1 - start_index
                                idx = i_step + 1 - start_index
                                if potential_type == 'linear':
                                    total_rewards.append(idx / len_trajectory)
                                if potential_type == 'triangular':
                                    total_rewards.append((idx * (idx + 1)) / (len_trajectory * (len_trajectory + 1)))
                                if potential_type == 'geometric':
                                    total_rewards.append((discount ** (len_trajectory - idx) - discount ** (len_trajectory - 1)) / (1 - discount ** (len_trajectory - 1)))
                                if potential_type == 'original_geometric':
                                    total_rewards.append(discount ** (len_trajectory - idx))
                                if potential_type == 'constant':
                                    total_rewards.append(1)
                            
                    # elif np.sum(rewards) < 100:
                    #     if potential_type == 'constant':
                    #         # failure trajectories with reward 0
                    #         start_index, end_index = 0, 100
                    #         for i_env_step in range(start_index, end_index):
                    #             state = states[i_env_step]
                    #             env_state = env_states[i_env_step]
                    #             next_state = next_states[i_env_step]
                    #             next_env_state = next_env_states[i_env_step]
                    #             total_env_states.append(env_state)
                    #             total_next_env_states.append(next_env_state)
                    #             total_rewards.append(0)
                
                for i in range(len(total_env_states)):
                    env_state = total_env_states[i]
                    reward = total_rewards[i]
                    next_env_state = total_next_env_states[i]
                    # only env_state, reward and next_env_state are useful
                    # agent.remember(state, agent_state, env_state, action, reward, next_state, next_agent_state, next_env_state, done, type='reverse')
                    agent.remember(state, agent_state, env_state, action, reward, next_state, next_agent_state, next_env_state, done, type='success')

                print('env {}: load {} trajectories {} transitions from {} to success replay buffer'.format(env_name, len(reward_trajectory_lengths), len(total_env_states), transition_path))
            else:
                print('{} does not exist.'.format(transition_path))

            update_start = time.time()
            for i_epoch in range(1000):
                if reward_model_type == 'reward':
                    forward_reward_loss = agent.update_reward(type='forward')
                if reward_model_type == 'potential':
                    forward_reward_loss = agent.update_potential(type='forward')
                # print(i_epoch, time.time()-update_start, forward_reward_loss)
            print('forward_reward_loss {} update {:.2f} seconds'.format(forward_reward_loss, time.time() - update_start))
                
    if use_reversed_reward:
        for i_env in range(num_envs):
            reward_trajectory_lengths = []
            total_env_states = []
            total_rewards = []
            total_next_env_states = []

            agent = agent1 if i_env == 0 else agent2
            env_name = env_names[i_env]
            reverse_env_name = env_names[(i_env+1)%num_envs]
            reverse_transition_path = 'generate/{}_transitions_{}trajectory_sparse.npy'.format(reverse_env_name, n_reverse_demo)                
            if os.path.exists(reverse_transition_path):
                transitions = np.load(reverse_transition_path, allow_pickle=True)

                for i_episode in range(n_reverse_demo):
                    states = []
                    env_states = []
                    rewards = []
                    next_states = []
                    next_env_states = []
                    for i, transition in enumerate(transitions[i_episode*horizon:(i_episode+1)*horizon]):
                        state, action, reward, next_state, done = transition
                        
                        goal_pos = []
                        # The trajectory is reversed, so the goal position should be changed accordingly.
                        if env_name == 'TwoArmPegInHole':
                            # the agent is for two arm peg in hole
                            # goal pos is set as the initial hole pos of the reversed trajectory
                            goal_pos = transitions[(i_episode+1)*horizon-1][3][10:13]
                        if env_name == 'TwoArmPegRemoval':
                            # this agent is for two arm peg removal
                            # goal pos is set as the last peg pos of the reversed trajectory 
                            goal_pos = transitions[i_episode*horizon][0][3:6]
                        if env_name == 'NutAssemblyRound':
                            # the agent is for nut assembly
                            goal_pos = np.array([0.1, -0.1, 0.85])
                        if env_name == 'NutDisAssemblyRound':
                            # the agent is for nut disassembly
                            goal_pos = np.array([-0.15, -0.15, 1.0])
                        if env_name == 'Stack':
                            # the agent is for block stack
                            # goal pos is set as the initial cubeB pos of the reversed trajectory
                            cubeB_x, cubeB_y = transitions[(i_episode+1)*horizon-1][3][6], transitions[(i_episode+1)*horizon-1][3][7]
                            goal_pos = np.array([cubeB_x, cubeB_y, 0.85])
                            init_cubeB_xy = transitions[(i_episode+1)*horizon-1][3][6:8]
                            goal_pos = np.concatenate([init_cubeB_xy, goal_pos])
                        if env_name == 'UnStack':
                            # the agent is for block unstack
                            # goal pos is set as the last cubeA pos of the reversed trajectory
                            goal_pos = transitions[i_episode*horizon][0][3:6]
                            init_cubeB_xy = transitions[(i_episode+1)*horizon-1][3][6:8]
                            goal_pos = np.concatenate([init_cubeB_xy, goal_pos])

                        state = change_goal_pos(state, env_name, goal_pos)
                        next_state = change_goal_pos(next_state, env_name, goal_pos)

                        agent_state, env_state = state2agentenv(state, env_name)
                        next_agent_state, next_env_state = state2agentenv(next_state, env_name)

                        states.append(state)
                        env_states.append(env_state)
                        rewards.append(reward)
                        next_states.append(next_state)
                        next_env_states.append(next_env_state)
                    if 1 in rewards and np.sum(rewards) > success_threshold:
                        first_reward = rewards.index(1)
                        
                        # env_state_diff = [np.linalg.norm(next_env_states[i]-env_states[i]) for i in range(first_reward)]
                        # first_env_change = list(np.array(env_state_diff) > 1e-3).index(True)
                        # start_index = max(first_env_change - 5, 0)

                        start_index = 0
                        end_index = first_reward
                        reward_trajectory_lengths.append(end_index - start_index)
                        print(start_index, end_index)

                        for i_step in range(start_index, end_index+1):
                            state = states[i_step]
                            env_state = env_states[i_step]
                            next_state = next_states[i_step]
                            next_env_state = next_env_states[i_step]
                            total_env_states.append(env_state)
                            if reward_model_type == 'reward':
                                total_rewards.append((1 - 1 / (end_index+1-start_index) * i_step) * reward_model_max_value)
                            if reward_model_type == 'potential':
                                len_trajectory = end_index + 1 - start_index
                                idx = i_step - start_index
                                if potential_type == 'linear':
                                    total_rewards.append(1 - idx / len_trajectory)
                                if potential_type == 'triangular':
                                    total_rewards.append(1 - (idx * (idx + 1)) / (len_trajectory * (len_trajectory + 1)))
                                if potential_type == 'geometric':
                                    total_rewards.append(1 - (discount ** (len_trajectory - idx) - discount ** (len_trajectory - 1)) / (1 - discount ** (len_trajectory - 1)))
                                if potential_type == 'original_geometric':
                                    total_rewards.append(discount ** idx)
                                if potential_type == 'constant':
                                    total_rewards.append(1)

                            total_next_env_states.append(next_env_state)

                for i in range(len(total_env_states)):
                    env_state = total_env_states[i]
                    reverse_reward = total_rewards[i]
                    next_env_state = total_next_env_states[i]
                    # only env_state, reward and next_env_state are useful
                    agent.remember(next_state, next_agent_state, next_env_state, action, reverse_reward, state, agent_state, env_state, done, type='reverse')
                    
                print('env {}: load {} trajectories {} transitions from {} to reverse replay buffer'.format(env_name, len(reward_trajectory_lengths), len(total_env_states), reverse_transition_path))
            else:
                print('{} does not exist.'.format(reverse_transition_path))

            update_start = time.time()
            for i_epoch in range(1000):
                if reward_model_type == 'reward':
                    reverse_reward_loss = agent.update_reward(type='reverse')
                if reward_model_type == 'potential':
                    reverse_reward_loss = agent.update_potential(type='reverse')
            print('reverse_reward_loss {} update {:.2f} seconds'.format(reverse_reward_loss, time.time() - update_start))

    score_history = []
    success_history = []
    i = 0
    num_training_episodes = 500
    eval_interval = 20
    save_video_interval = eval_interval
    save_model_interval = 500
    num_eval = 20 #20
    num_sampling_episodes = 5 #5
    START = time.time()
    while i < num_training_episodes:
        # the agent will interact with the environment
        if i % eval_interval == 0: #0
            #test
            for i_test_env in range(num_envs):
                # if i_test_env == 0: continue
                env_name = env_names[i_test_env]
                eval_df_path = eval_df_paths[i_test_env]
                video_dir = video_dirs[i_test_env]
                test_env = test_env1 if i_test_env == 0 else test_env2
                agent = agent1 if i_test_env == 0 else agent2
                eval_df = eval_df1 if i_test_env == 0 else eval_df2
                
                start = time.time()   
                i_test = 0
            
                while i_test < num_eval:
                    done = False
                    test_score = 0
                    env_state_this_episode = []
                    next_env_state_this_episode = []
                    obs = test_env.reset()
                    state = obs2state(obs, test_env, env_name)
                    i_step = 0
                    if save_video and i % save_video_interval == 0:
                        frames = []
                    while not done:
                        if save_video and i % save_video_interval == 0:
                            frame = np.concatenate([obs['agentview_image'], obs['sideview_image']])[::-1]
                            frames.append(frame)
                        i_step += 1
                        action = agent.act(state, sample=False)

                        if env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
                            step_action = np.concatenate([action, np.zeros(6)])
                        else:
                            step_action = action

                        try:
                            next_obs, reward, done, _ = test_env.step(step_action)
                        except MujocoException as e:
                            print('got MujocoException {} at test episode {} timestep {}'.format(str(e), i_test, i_step))
                            print('state {} action {}'.format(state, action))
                            done = True
                        # earlystop for sparse reward setting
                        # if test_env._check_success() and not reward_shaping: done = True
                        next_state = obs2state(next_obs, test_env, env_name)
                        test_score += reward
                        _, env_state = state2agentenv(state, env_name)
                        _, next_env_state = state2agentenv(next_state, env_name)
                        env_state_this_episode.append(env_state)
                        next_env_state_this_episode.append(next_env_state)
                        
                        obs = next_obs
                        state = next_state
                    
                    with torch.no_grad():
                        if reward_model_type == 'reward':
                            forward_score = agent.forward_reward(torch.FloatTensor(env_state_this_episode).to(agent.device), torch.FloatTensor(next_env_state_this_episode).to(agent.device)).sum().cpu().numpy()
                            reverse_score = agent.reverse_reward(torch.FloatTensor(env_state_this_episode).to(agent.device), torch.FloatTensor(next_env_state_this_episode).to(agent.device)).sum().cpu().numpy()
                        if reward_model_type == 'potential':
                            forward_new_potential = agent.forward_potential(torch.FloatTensor(next_env_state_this_episode).to(agent.device))
                            forward_potential = agent.forward_potential(torch.FloatTensor(env_state_this_episode).to(agent.device))
                            reverse_new_potential = agent.reverse_potential(torch.FloatTensor(next_env_state_this_episode).to(agent.device))
                            reverse_potential = agent.reverse_potential(torch.FloatTensor(env_state_this_episode).to(agent.device))
                            forward_score = torch.clip((forward_new_potential - forward_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value).sum().cpu().numpy()
                            reverse_score = torch.clip((reverse_new_potential - reverse_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value).sum().cpu().numpy()

                        estimated_test_score = 0
                        if use_forward_reward: estimated_test_score += forward_score
                        if use_reversed_reward: estimated_test_score += reverse_score
                        if use_forward_reward and use_reversed_reward: estimated_test_score /= 2
                        
                    test_success = int(test_env._check_success())
                    print('env {} test at episode {}; i_test {}; num_steps {}; score {:.2f}; estimated score {:.2f}; success {}; time {:.2f}; total_time {:.2f}'.format(env_name, i, i_test, i_step, test_score, estimated_test_score, test_success, time.time()-start, time.time()-START))
                    eval_df.loc[len(eval_df.index)] = [i, i_test, i_step, test_score, estimated_test_score, test_success, time.time()-start, time.time()-START]
                    eval_df.to_csv(eval_df_path)

                    if save_video and i % save_video_interval == 0:
                        video_start = time.time()
                        test_video_path = os.path.join(video_dir, 'episode{}_test{}.mp4'.format(i, i_test))
                        imageio.mimsave(uri=test_video_path, ims=frames, fps=20, macro_block_size = 1)
                        print('video saved at {} in {:.3f}s'.format(test_video_path, time.time()-video_start))

                    i_test += 1
        
        # train
        for i_env in range(num_envs):
            # if i_env == 0: continue
            env_name = env_names[i_env]
            reverse_env_name = env_names[(i_env+1)%num_envs]
            train_df_path = train_df_paths[i_env]
            model_dir = model_dirs[i_env]
            train_video_dir = train_video_dirs[i_env]
            replay_buffer_dir = replay_buffer_dirs[i_env]
            env = env1 if i_env == 0 else env2
            agent = agent1 if i_env == 0 else agent2
            reverse_agent = agent1 if i_env == 1 else agent2
            train_df = train_df1 if i_env == 0 else train_df2
            
            start = time.time()
            done = False
            score = 0
            obs = env.reset()
            state = obs2state(obs, env, env_name)
            i_step = 0
            train_time = 0
            actor_losses = []
            alpha_losses = []
            critic_losses = []
            forward_reward_losses = []
            reverse_reward_losses = []
            reversible_ratios = []
            for_losses = []
            inv_losses = []
            alpha = 0
            log_probs = []

            state_this_episode = []
            agent_state_this_episode = []
            env_state_this_episode = []
            act_this_episode = []
            reward_this_episode = []
            next_state_this_episode = []
            next_agent_state_this_episode = []
            next_env_state_this_episode = []
            done_no_max_this_episode = []

            if save_video_train:
                frames = []
            while not done:
                if save_video_train:
                    frame = np.concatenate([obs['agentview_image'], obs['sideview_image']])[::-1]
                    frames.append(frame)
                i_step += 1
                if i < num_sampling_episodes:
                    # choose actions randomly
                    action = np.random.uniform(action_low, action_high)
                else:
                    # choose actions based on the actor
                    action = agent.act(state, sample=True)

                if env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
                    step_action = np.concatenate([action, np.zeros(6)])
                else:
                    step_action = action
               
                try:
                    next_obs, reward, done, _ = env.step(step_action)
                except MujocoException as e:
                    print('got MujocoException {} at episode {} timestep {}'.format(str(e), i, i_step))
                    print('state {} action {}'.format(state, action))
                    done = True
                done_no_max = 0.0 if i_step == _max_episode_steps else float(done)
                next_state = obs2state(next_obs, env, env_name)

                agent_state, env_state = state2agentenv(state, env_name)
                next_agent_state, next_env_state = state2agentenv(next_state, env_name)

                state_this_episode.append(state)
                agent_state_this_episode.append(agent_state)
                env_state_this_episode.append(env_state)
                act_this_episode.append(action)
                reward_this_episode.append(reward)
                next_state_this_episode.append(next_state)
                next_agent_state_this_episode.append(next_agent_state)
                next_env_state_this_episode.append(next_env_state)
                done_no_max_this_episode.append(done_no_max)

                score += reward
                
                # agent updates its parameters
                if agent.replay_buffer.mem_cntr > agent.batch_size and i>=num_sampling_episodes:
                    train_start = time.time()

                    forward_dynamics = agent1.forward_dynamics
                    inverse_dynamics = agent1.inverse_dynamics
                    filter_transition = utils.filter_transition
                    # update agent
                    critic_loss, actor_loss, alpha_loss, forward_reward_loss, reverse_reward_loss, reversible_ratio, alpha, log_prob = agent.update(i_step, use_forward_reward, use_reversed_reward, reward_model_type, reward_model_max_value, horizon, use_reversed_transition, forward_dynamics, inverse_dynamics, thresholds, state_max[i_env], state_min[i_env], filter_transition, filter_type)
                    # print('step {} step_time {:.5f} episode_time {:.2f}'.format(i_step, time.time()-train_start, time.time()-start))

                    for_loss, inv_loss = 0, 0
                    if i_env == 0:
                        for_loss = agent.update_forward_dynamics(agent1, agent2)
                        inv_loss = agent.update_inverse_dynamics(agent1, agent2)

                    if type(log_prob) != int:
                        log_probs.append(torch.mean(log_prob).cpu().detach().numpy())
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
                    alpha_losses.append(alpha_loss)
                    forward_reward_losses.append(forward_reward_loss)
                    reverse_reward_losses.append(reverse_reward_loss)
                    reversible_ratios.append(reversible_ratio)
                    for_losses.append(for_loss)
                    inv_losses.append(inv_loss)
                    
                    train_time += time.time() - train_start

                obs, state, agent_state, env_state = next_obs, next_state, next_agent_state, next_env_state
            
            # store transitions of this episode to the replay buffer
            for i_sample in range(len(state_this_episode)):
                state, agent_state, env_state, act, reward, next_state, next_agent_state, next_env_state, done_no_max = state_this_episode[i_sample], agent_state_this_episode[i_sample], env_state_this_episode[i_sample], act_this_episode[i_sample], reward_this_episode[i_sample], next_state_this_episode[i_sample], next_agent_state_this_episode[i_sample], next_env_state_this_episode[i_sample], done_no_max_this_episode[i_sample]
                agent.remember(state, agent_state, env_state, act, reward, next_state, next_agent_state, next_env_state, done_no_max)

            reverse_rewards = [0]
            if use_reversed_transition:
                # reverse this trajectory
                reverse_states = next_state_this_episode.copy()
                reverse_next_states = state_this_episode.copy()                
                info = {}

                # The trajectory is reversed, so the goal position should be changed accordingly
                if reverse_env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval']:
                    if reverse_env_name == 'TwoArmPegInHole': 
                        # the reverse agent is for two arm peg in hole
                        # goal pos is set as the initial hole pos of the reversed trajectory
                        goal_pos = next_state_this_episode[-1][10:13]
                    if reverse_env_name == 'TwoArmPegRemoval':
                        # the reverse agent is for two arm peg removal
                        # goal pos is set as the last peg pos of the reversed trajectory 
                        goal_pos = state_this_episode[0][3:6]
                    # hole init pos is set as the initial hole pos of the reversed trajectory
                    info['hole_init_pos'] = next_state_this_episode[-1][10:13]
                if reverse_env_name in ['NutAssemblyRound', 'NutDisAssemblyRound']:
                    if reverse_env_name == 'NutAssemblyRound':
                        # the reverse agent is for nut assembly
                        goal_pos = np.array([0.1, -0.1, 0.85])
                    if reverse_env_name == 'NutDisAssemblyRound':
                        # the reverse agent is for nut disassembly
                        goal_pos = np.array([-0.15, -0.15, 1.0])
                if reverse_env_name in ['Stack', 'UnStack']:
                    if reverse_env_name == 'Stack':
                        # the reverse agent is for block stack
                        # goal pos is set as the initial cubeB pos of the reversed trajectory
                        cubeB_x, cubeB_y = next_state_this_episode[-1][6], next_state_this_episode[-1][7]
                        goal_pos = np.array([cubeB_x, cubeB_y, 0.85])
                    if reverse_env_name == 'UnStack':
                        # the reverse agent is for block unstack
                        # goal pos is set as the last cubeA pos of the reversed trajectory
                        goal_pos = state_this_episode[0][3:6]
                    init_cubeB_xy = next_state_this_episode[-1][6:8]
                    goal_pos = np.concatenate([init_cubeB_xy, goal_pos])

                if reverse_env_name in ['TwoArmPegInHole', 'TwoArmPegRemoval', 'NutAssemblyRound', 'NutDisAssemblyRound', 'Stack', 'UnStack']:
                    reverse_states = [change_goal_pos(s, reverse_env_name, goal_pos) for s in reverse_states]
                    reverse_next_states = [change_goal_pos(s, reverse_env_name, goal_pos) for s in reverse_next_states]

                reverse_rewards = [state_reward(s, reverse_env_name, reward_shaping, info) for s in reverse_next_states]
                # add high value trajectories to reverse agent replay buffer
                if np.sum(reverse_rewards) > 0:
                    for i_sample in range(len(reverse_states)):
                        reverse_state, act, reverse_reward, reverse_next_state, done_no_max = reverse_states[i_sample], act_this_episode[i_sample], reverse_rewards[i_sample], reverse_next_states[i_sample], done_no_max_this_episode[i_sample]

                        reverse_agent_state, reverse_env_state = state2agentenv(reverse_state, reverse_env_name)
                        reverse_next_agent_state, reverse_next_env_state = state2agentenv(reverse_next_state, reverse_env_name)

                        reverse_agent.remember(reverse_state, reverse_agent_state, reverse_env_state, act, reverse_reward, reverse_next_state, reverse_next_agent_state, reverse_next_env_state, done_no_max, type='reversal')

            score_history.append(score)
            success = int(env._check_success())
            success_history.append(success)

            with torch.no_grad():
                if reward_model_type == 'reward':
                    forward_score = agent.forward_reward(torch.FloatTensor(env_state_this_episode).to(agent.device), torch.FloatTensor(next_env_state_this_episode).to(agent.device)).sum().cpu().numpy()
                    reverse_score = agent.reverse_reward(torch.FloatTensor(env_state_this_episode).to(agent.device), torch.FloatTensor(next_env_state_this_episode).to(agent.device)).sum().cpu().numpy()
                if reward_model_type == 'potential':
                    forward_new_potential = agent.forward_potential(torch.FloatTensor(next_env_state_this_episode).to(agent.device))
                    forward_potential = agent.forward_potential(torch.FloatTensor(env_state_this_episode).to(agent.device))
                    reverse_new_potential = agent.reverse_potential(torch.FloatTensor(next_env_state_this_episode).to(agent.device))
                    reverse_potential = agent.reverse_potential(torch.FloatTensor(env_state_this_episode).to(agent.device))
                    forward_score = torch.clip((forward_new_potential - forward_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value).sum().cpu().numpy()
                    reverse_score = torch.clip((reverse_new_potential - reverse_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value).sum().cpu().numpy()

                estimated_score = 0
                if use_forward_reward: estimated_score += forward_score
                if use_reversed_reward: estimated_score += reverse_score
                if use_forward_reward and use_reversed_reward: estimated_score /= 2
                
            if use_forward_reward:
                if success == 1 and np.sum(reward_this_episode) > success_threshold:                    
                    first_reward = reward_this_episode.index(1)
                    # env_state_diff = [np.linalg.norm(next_env_state_this_episode[i]-env_state_this_episode[i]) for i in range(first_reward)]
                    # first_env_change = list(np.array(env_state_diff) > 1e-3).index(True)
                    # start_index = max(first_env_change - 5, 0)

                    start_index = 0
                    end_index = first_reward
                    print(start_index, end_index)
                    
                    for i_env_step in range(start_index, end_index+1):
                        env_state = env_state_this_episode[i_env_step]
                        next_env_state = next_env_state_this_episode[i_env_step]
                        if reward_model_type == 'reward':
                            reward = (1 / (end_index+1) * (i_env_step + 1)) * reward_model_max_value
                        if reward_model_type == 'potential':
                            len_trajectory = end_index + 1 - start_index
                            idx = i_env_step + 1 - start_index
                            if potential_type == 'linear':
                                reward = idx / len_trajectory
                            if potential_type == 'triangular':
                                reward = (idx * (idx + 1)) / (len_trajectory * (len_trajectory + 1))
                            if potential_type == 'geometric':
                                reward = (discount ** (len_trajectory - idx) - discount ** (len_trajectory - 1)) / (1 - discount ** (len_trajectory - 1))
                            if potential_type == 'original_geometric':
                                reward = discount ** (len_trajectory - idx)
                            if potential_type == 'constant':
                                reward = 1
                        # only env_state, reward and next_env_state are useful
                        agent.remember(state, agent_state, env_state, action, reward, next_state, agent_state, next_env_state, done, type='success')       
                            
                    for i_epoch in range(100): 
                        if reward_model_type == 'reward':
                            forward_reward_loss = agent.update_reward(type='forward')
                        if reward_model_type == 'potential':
                            forward_reward_loss = agent.update_potential(type='forward')
                
                # elif success == 0 and np.sum(reward_this_episode) < 100:
                #     if potential_type == 'constant':
                #         # failure trajectories with reward 0
                #         start_index, end_index = 0, 100
                #         for i_env_step in range(start_index, end_index):
                #             env_state = env_state_this_episode[i_env_step]
                #             next_env_state = next_env_state_this_episode[i_env_step]
                #             reward = 0
                #             # only env_state, reward and next_env_state are useful
                #             agent.remember(state, agent_state, env_state, action, reward, next_state, agent_state, next_env_state, done, type='success')
                                
                #         for i_epoch in range(100): 
                #             forward_reward_loss = agent.update_potential(type='forward')
            
            if use_reversed_reward:
                if success == 1 and np.sum(reward_this_episode) > success_threshold:
                    first_reward = reward_this_episode.index(1)
                    start_index = 0
                    end_index = first_reward
                    print(start_index, end_index)

                    for i_env_step in range(start_index, end_index+1):
                        state = state_this_episode[i_env_step]
                        next_state = next_state_this_episode[i_env_step]
                        # The trajectory is reversed, so the goal position should be changed accordingly.
                        goal_pos = []
                        if reverse_env_name == 'TwoArmPegInHole':
                            # the reverse agent is for two arm peg in hole
                            # goal pos is set as the initial hole pos of the reversed trajectory
                            goal_pos = next_state_this_episode[end_index+1][10:13]
                        if reverse_env_name == 'TwoArmPegRemoval':
                            # the reverse agent is for two arm peg removal
                            # goal pos is set as the last peg pos of the reversed trajectory 
                            goal_pos = state_this_episode[0][3:6]
                        if reverse_env_name == 'NutAssemblyRound':
                            # the reverse agent is for nut assembly
                            goal_pos = np.array([0.1, -0.1, 0.85])
                        if reverse_env_name == 'NutDisAssemblyRound':
                            # the reverse agent is for nut disassembly
                            goal_pos = np.array([-0.15, -0.15, 1.0])
                        if reverse_env_name == 'Stack':
                            # the reverse agent is for block stack
                            # goal pos is set as the initial cubeB pos of the reversed trajectory
                            cubeB_x, cubeB_y = next_state_this_episode[end_index+1][6], next_state_this_episode[end_index+1][7]
                            goal_pos = np.array([cubeB_x, cubeB_y, 0.85])
                            init_cubeB_xy = next_state_this_episode[end_index+1][6:8]
                            goal_pos = np.concatenate([init_cubeB_xy, goal_pos])
                        if reverse_env_name == 'UnStack':
                            # the reverse agent is for block unstack
                            # goal pos is set as the last cubeA pos of the reversed trajectory
                            goal_pos = state_this_episode[0][3:6]
                            init_cubeB_xy = next_state_this_episode[end_index+1][6:8]
                            goal_pos = np.concatenate([init_cubeB_xy, goal_pos])

                        state = change_goal_pos(state, reverse_env_name, goal_pos)
                        next_state = change_goal_pos(next_state, reverse_env_name, goal_pos)

                        _, env_state = state2agentenv(state, reverse_env_name)
                        _, next_env_state = state2agentenv(next_state, reverse_env_name)

                        if reward_model_type == 'reward':
                            reverse_reward = (1 - 1 / (end_index+1) * i_env_step) * reward_model_max_value
                        if reward_model_type == 'potential':
                            len_trajectory = end_index + 1 - start_index
                            idx = i_env_step - start_index
                            if potential_type == 'linear':
                                reverse_reward = 1 - idx / len_trajectory
                            if potential_type == 'triangular':
                                reverse_reward = 1 - (idx * (idx + 1)) / (len_trajectory * (len_trajectory + 1))
                            if potential_type == 'geometric':
                                reverse_reward = 1 - (discount ** (len_trajectory - idx) - discount ** (len_trajectory - 1)) / (1 - discount ** (len_trajectory - 1))
                            if potential_type == 'original_geometric':
                                reverse_reward = discount ** idx
                            if potential_type == 'constant':
                                reverse_reward = 1
                        reverse_agent.remember(next_state, agent_state, next_env_state, action, reverse_reward, state, agent_state, env_state, done, type='reverse')

                    for i_epoch in range(100):
                        if reward_model_type == 'reward':
                            reverse_reward_loss = reverse_agent.update_reward(type='reverse')
                        if reward_model_type == 'potential':
                            reverse_reward_loss = reverse_agent.update_potential(type='reverse')

                        
            print('env {}; train at episode {}; num_steps {}; train score {:.2f}; 100game avg score {:.2f}; estimated_score {:.3f}; success {}; 100 game avg success {:.2f}; critic_loss {:.5f}; actor_loss {:.5f}; forward_reward_loss {:.5f}; reverse_reward_loss {:.5f}; reversible_ratio {:.5f} for_loss {:.5f}; inv_loss {:.5f}; alpha_loss {:.5f}; alpha {:.5f}; log_prob {:.5f}; env_buffer {}; demo_buffer {}; success_buffer {}; reverse_buffer {}; reversal_buffer {}; train_time {:.3f}; episode_time {:.2f}; total_time {:.2f}'.format(env_name, i, i_step, score, np.mean(score_history[-100:]), estimated_score, success, np.mean(success_history[-100:]), np.mean(critic_losses), np.mean(actor_losses), np.mean(forward_reward_losses), np.mean(reverse_reward_losses), np.mean(reversible_ratios), np.mean(for_losses), np.mean(inv_losses), np.mean(alpha_losses), alpha, np.mean(log_probs), agent.replay_buffer.mem_cntr, agent.demo_replay_buffer.mem_cntr, agent.success_replay_buffer.mem_cntr, agent.reverse_replay_buffer.mem_cntr, agent.reversal_replay_buffer.mem_cntr, train_time, time.time()-start, time.time()-START))

            train_df.loc[len(train_df.index)] = [i, i_step, score, success, estimated_score, np.mean(critic_losses), np.mean(actor_losses), np.mean(forward_reward_losses), np.mean(reverse_reward_losses), np.mean(reversible_ratios), np.mean(for_losses), np.mean(inv_losses), np.mean(alpha_losses), alpha, np.mean(log_probs), agent.replay_buffer.mem_cntr, agent.demo_replay_buffer.mem_cntr, agent.success_replay_buffer.mem_cntr, agent.reverse_replay_buffer.mem_cntr, agent.reversal_replay_buffer.mem_cntr, train_time, time.time()-start, time.time()-START]
            train_df.to_csv(train_df_path)

            if save_video_train:
                video_start = time.time()
                train_video_path = os.path.join(train_video_dir, 'train_episode{}.mp4'.format(i))
                imageio.mimsave(uri=train_video_path, ims=frames, fps=20, macro_block_size = 1)
                print('video saved at {} in {:.3f}s'.format(train_video_path, time.time()-video_start))
            
        i += 1