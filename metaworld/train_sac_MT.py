import copy
import math
from mujoco_py.builder import MujocoException
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import obs2state, state2agentenv, state_reward, change_target_pos
import argparse
import imageio
import pandas as pd
from sac_MT import SACAgent
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

import warnings 
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, help = 'batch_size', default = 128) # 128 for MT10 (1280), 32 for MT50 (1600)
    parser.add_argument('--hidden_depth', type = int, help = 'hidden_depth', default = 3)
    parser.add_argument('--hidden_dim', type = int, help = 'hidden_dim', default = 400)
    parser.add_argument('--seed', type = int, help = 'seed', default = 1)
    parser.add_argument('--num_envs', type = int, help = 'num_envs', default = 10) # 10, 50
    parser.add_argument('--n_demo', type = int, help = 'n_demo', default = 10)
    parser.add_argument('--reward_shaping', type=str, default='true')

    parser.add_argument('--use_forward_reward', type=str, default = 'false')
    parser.add_argument('--use_reversed_reward', type=str, default = 'false')
    parser.add_argument('--n_forward_demo', type = int, help = 'n_forward_demo', default = 10)
    parser.add_argument('--n_reverse_demo', type = int, help = 'n_reverse_demo', default = 10)
    parser.add_argument('--reward_model_type', type=str, default = 'reward') # potential
    parser.add_argument('--potential_type', type=str, default = 'linear') # linear

    parser.add_argument('--use_reversed_transition', type=str, default = 'false')
    parser.add_argument('--diff_threshold', type=float, default = 0.01)
    parser.add_argument('--filter_type', type=str, default = 'None') # None, state_max_diff

    args = vars(parser.parse_args())
    batch_size = args['batch_size']
    hidden_depth = args['hidden_depth']
    hidden_dim = args['hidden_dim']
    print('batch_size {} hidden_depth {} hidden_dim {}'.format(batch_size, hidden_depth, hidden_dim))

    seed = args['seed']
    print('seed {}'.format(seed))
    utils.set_seed_everywhere(seed)
    num_envs = args['num_envs']
    print('num_envs', num_envs)
    n_demo = args['n_demo']
    print('n_demo', n_demo)
    reward_shaping = True if args['reward_shaping'] == 'true' else False
    print('reward_shaping {}'.format(reward_shaping))
    
    use_forward_reward = False if args['use_forward_reward'] == 'false' else True
    print('use_forward_reward', use_forward_reward)
    use_reversed_reward = False if args['use_reversed_reward'] == 'false' else True
    print('use_reversed_reward', use_reversed_reward)
    n_forward_demo = args['n_forward_demo']
    print('n_forward_demo', n_forward_demo)
    n_reverse_demo = args['n_reverse_demo']
    print('n_reverse_demo', n_reverse_demo)
    reward_model_type = args['reward_model_type']
    print('reward_model_type {}'.format(reward_model_type))
    potential_type = args['potential_type']
    print('potential_type {}'.format(potential_type))
    
    use_reversed_transition = False if args['use_reversed_transition'] == 'false' else True
    print('use_reversed_transition', use_reversed_transition)
    diff_threshold = args['diff_threshold']
    print('diff_threshold {}'.format(diff_threshold))
    all_thresholds = []
    for _ in range(num_envs): all_thresholds.append([diff_threshold])
    print('all_thresholds {}'.format(all_thresholds))
    filter_type = args['filter_type']
    print('filter_type {}'.format(filter_type))
    
    save_video = not True
    print('save_video {}'.format(save_video))
    save_video_train = not True
    print('save_video_train {}'.format(save_video_train))

    algo_name = 'SAC_MT{}_{}demo'.format(num_envs, n_demo)
    if use_forward_reward and use_reversed_reward: algo_name += '_use_reversed_reward_{}reverse_demo_use_forward_reward_{}demo_separate'.format(n_reverse_demo, n_forward_demo)
    if reward_model_type != 'reward': algo_name += '_{}_model'.format(reward_model_type)
    if reward_model_type == 'potential': algo_name += '_{}_potential'.format(potential_type)
    if use_reversed_transition:
        if filter_type == 'None':
            algo_name += '_use_reversed_transition'
        if filter_type == 'state_max_diff':
            algo_name += '_use_reversed_transition_state_max_diff{}'.format(diff_threshold)
    if not reward_shaping: algo_name += '_sparse'
    print('algo_name: {} state'.format(algo_name))

    reward_model_max_value = 0.5
    success_threshold = 50
    env_embedding_size = 50

    if num_envs == 10:
        env_names = ['button-press', 'door-open', 'drawer-open', 'drawer-close', 'peg-insert-side', 'pick-place', 'push', 'reach', 'window-open', 'window-close']
        reversible_env_pairs = [['drawer-open', 'drawer-close'], ['window-open', 'window-close']]
        self_reversible_envs = ['reach', 'pick-place']
        
        reversible_envs = [env_name for env_pair in reversible_env_pairs for env_name in env_pair]
        reversible_flags = [env_name in reversible_envs for env_name in env_names]
        self_reversible_flags = [env_name in self_reversible_envs for env_name in env_names]
        
        reversible_dict = {}
        for reversible_env_pair in reversible_env_pairs:
            env1_name, env2_name = reversible_env_pair
            env1_index, env2_index = env_names.index(env1_name), env_names.index(env2_name)
            reversible_dict[env1_index], reversible_dict[env2_index] = env2_index, env1_index
        for self_reversible_env in self_reversible_envs:
            env_index = env_names.index(self_reversible_env)
            reversible_dict[env_index] = env_index

        env_dynamics_indices = [_ for _ in range(num_envs)]
        for i_env in range(num_envs):
            if env_names[i_env] in reversible_envs:
                reverse_i_env = reversible_dict[i_env]
                env_dynamics_indices[i_env] = min(i_env, reverse_i_env)

        # reversible_flags = [False, False, True, True, False, False, False, False, True, True]
        # self_reversible_flags = [False, False, False, False, False, True, False, True, False, False]
        # env_dynamics_indices = [0, 1, 2, 2, 4, 5, 6, 7, 8, 8]
        # reversible_dict = {}
        # reversible_dict[2] = 3
        # reversible_dict[3] = 2
        # reversible_dict[5] = 5
        # reversible_dict[7] = 7
        # reversible_dict[8] = 9
        # reversible_dict[9] = 8

    if num_envs == 50:
        env_names = ['assembly', 'basketball', 'bin-picking', 'box-close', 'button-press', 'button-press-wall', 'button-press-topdown', 'button-press-topdown-wall', 'dial-turn', 'disassemble', 'coffee-button', 'coffee-pull', 'coffee-push', 'door-lock', 'door-unlock', 'door-open', 'door-close', 'drawer-open', 'drawer-close', 'faucet-open', 'faucet-close', 'hammer', 'hand-insert', 'handle-press', 'handle-pull', 'handle-press-side', 'handle-pull-side', 'lever-pull', 'peg-insert-side', 'peg-unplug-side', 'pick-out-of-hole', 'pick-place', 'pick-place-wall', 'plate-slide', 'plate-slide-back', 'plate-slide-side', 'plate-slide-back-side', 'push', 'push-back', 'push-wall', 'reach', 'reach-wall', 'shelf-place', 'soccer', 'stick-push', 'stick-pull', 'sweep', 'sweep-into', 'window-open', 'window-close']
        reversible_env_pairs = [['assembly', 'disassemble'], ['coffee-pull', 'coffee-push'], ['door-open', 'door-close'], ['door-lock', 'door-unlock'], ['drawer-open', 'drawer-close'], ['faucet-open', 'faucet-close'], ['handle-press', 'handle-pull'], ['peg-insert-side', 'peg-unplug-side'], ['plate-slide', 'plate-slide-back'], ['plate-slide-side', 'plate-slide-back-side'], ['push', 'push-back'], ['window-open', 'window-close']]
        self_reversible_envs = ['reach', 'reach-wall', 'pick-place', 'pick-place-wall']       
        
        reversible_envs = [env_name for env_pair in reversible_env_pairs for env_name in env_pair]
        reversible_flags = [env_name in reversible_envs for env_name in env_names]
        self_reversible_flags = [env_name in self_reversible_envs for env_name in env_names]
        
        reversible_dict = {}
        for reversible_env_pair in reversible_env_pairs:
            env1_name, env2_name = reversible_env_pair
            env1_index, env2_index = env_names.index(env1_name), env_names.index(env2_name)
            reversible_dict[env1_index], reversible_dict[env2_index] = env2_index, env1_index
        for self_reversible_env in self_reversible_envs:
            env_index = env_names.index(self_reversible_env)
            reversible_dict[env_index] = env_index

        env_dynamics_indices = [_ for _ in range(num_envs)]
        for i_env in range(num_envs):
            if env_names[i_env] in reversible_envs:
                reverse_i_env = reversible_dict[i_env]
                env_dynamics_indices[i_env] = min(i_env, reverse_i_env)

    all_env_embeddings = np.load('MT50_task_embedding.npy', allow_pickle=True).item()
    env_embeddings = {}
    for i_env in range(num_envs):
        env_name = env_names[i_env]
        env_embeddings[i_env] = all_env_embeddings[env_name]

    # Get action size, action limits and state_size
    action_size = 4
    action_low, action_high = np.ones(action_size) * -1, np.ones(action_size)
    action_shape = (action_size,)
    action_range = [float(action_low.min()), float(action_high.max())]
    state_size, agent_state_size, env_state_size = 21, 7, 14

    # initialize the tasks
    horizon = _max_episode_steps = 200
    all_envs = {}
    all_test_envs = {}
    all_goal_pos = {}
    for i_env in range(num_envs):
        env_name = env_names[i_env]
        env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["{}-v2-goal-hidden".format(env_name)](seed=seed)
        env.render_mode = 'rgb_array'
        test_env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["{}-v2-goal-hidden".format(env_name)](seed=seed)
        test_env.render_mode = 'rgb_array'

        env.max_path_length = test_env.max_path_length = horizon

        all_envs[i_env] = env
        all_test_envs[i_env] = test_env

        obs, _ = all_envs[i_env].reset()
        action = np.zeros(action_size)
        state = obs2state(obs, action, env, env_name)
        all_goal_pos[i_env] = state[14:17]
    
    all_state_max = {}
    all_state_min = {}
    for i_env in range(num_envs):
        env_name = env_names[i_env]
        transition_path = 'generate/{}_transitions_{}trajectory_sparse.npy'.format(env_name, n_demo)
        if os.path.exists(transition_path):
            transitions = np.load(transition_path, allow_pickle=True)
            state = transitions[:, 0]
            state_size = len(state[0])
            state = np.concatenate(state, axis=0).reshape(-1, state_size)
            state_max = np.max(state, axis=0) 
            state_min = np.min(state, axis=0)

            all_state_max[i_env] = state_max
            all_state_min[i_env] = state_min
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    discount, init_temperature = 0.99, 0.1
    actor_lr, critic_lr, reward_lr, potential_lr, for_lr, inv_lr, alpha_lr = 3e-4, 3e-4, 3e-4, 3e-4, 3e-4, 3e-4, 3e-4
    print('actor_lr {} critic_lr {} reward_lr {} potential_lr {} for_lr {} inv_lr {} alpha_lr {}'.format(actor_lr, critic_lr, reward_lr, potential_lr, for_lr, inv_lr, alpha_lr))
    actor_update_frequency, critic_tau, critic_target_update_frequency = 2, 0.005, 1
    log_std_bounds = [-20, 2]
    num_training_episodes = 500
    if num_envs == 50:
        num_training_episodes = 250
    num_sampling_episodes = 10
    replay_buffer_size, demo_replay_buffer_size, reverse_replay_buffer_size = num_training_episodes * horizon, n_demo * horizon, num_training_episodes * horizon
    print('replay_buffer_size {} demo_replay_buffer_size {} reverse_replay_buffer_size {}'.format(replay_buffer_size, demo_replay_buffer_size, reverse_replay_buffer_size))

    agent = SACAgent(state_size, agent_state_size, env_state_size, action_size, action_range, device, discount, init_temperature, actor_lr, critic_lr, potential_lr, for_lr, inv_lr, alpha_lr, actor_update_frequency, critic_tau, critic_target_update_frequency, batch_size, hidden_dim, hidden_depth, log_std_bounds, demo_replay_buffer_size, replay_buffer_size, num_envs, env_names, env_embedding_size, env_embeddings, env_dynamics_indices)

    log_dirs = {}
    train_video_dirs = {}
    video_dirs = {}
    train_dfs = {}
    train_df_paths = {}
    eval_dfs = {}
    eval_df_paths = {}
    for i_env in range(num_envs):
        env_name = env_names[i_env]
        log_dir = 'runs/{}/{}_seed{}/'.format(algo_name, env_name, seed)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_video_dir = os.path.join(log_dir, 'train_videos')
        if not os.path.exists(train_video_dir):
            os.mkdir(train_video_dir)
        video_dir = os.path.join(log_dir, 'videos')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)

        train_df = pd.DataFrame(columns=['episode', 'num_steps', 'score', 'success', 'estimated_score', 'critic_loss', 'actor_loss', 'forward_reward_loss', 'reverse_reward_loss', 'reversible_ratio', 'for_loss', 'inv_loss', 'alpha_loss', 'alpha', 'log_prob', 'env_buffer', 'demo_buffer', 'success_buffer', 'reverse_buffer', 'reversal_buffer', 'train_time', 'episode_time', 'total_time'])
        train_df_path = os.path.join(log_dir, 'train.csv')
        eval_df = pd.DataFrame(columns=['episode', 'i_test', 'num_steps', 'eval_score', 'estimated_eval_score', 'eval_success', 'episode_time', 'total_time'])
        eval_df_path = os.path.join(log_dir, 'eval.csv')

        log_dirs[i_env] = log_dir
        train_video_dirs[i_env] = train_video_dir
        video_dirs[i_env] = video_dir
        train_dfs[i_env] = train_df
        train_df_paths[i_env] = train_df_path
        eval_dfs[i_env] = eval_df
        eval_df_paths[i_env] = eval_df_path

    # store forward and reverse demo transitions
    for i_env in range(num_envs):
        print('-'*20)
        # store forward demo transitions
        demo_replay_buffer = agent.demo_replay_buffers[i_env]
        env_name = env_names[i_env]
        transition_path = 'generate/{}_transitions_{}trajectory_sparse.npy'.format(env_name, n_demo)
        if os.path.exists(transition_path):
            transitions = np.load(transition_path, allow_pickle=True)
            for transition in transitions:
                state, action, reward, next_state, done = transition
                agent_state, env_state = state2agentenv(state, env_name)
                next_agent_state, next_env_state = state2agentenv(next_state, env_name)
                demo_replay_buffer.store_transition(state, agent_state, env_state, action, reward, next_state, next_agent_state, next_env_state, done)
            
            print('env {}: load {} transitions from {} to demo replay buffer'.format(env_name, len(transitions), transition_path))
            print('demo reward density {:.4f}'.format(np.mean(demo_replay_buffer.reward_memory[:demo_replay_buffer.mem_cntr])))
        else:
            print('{} does not exist.'.format(transition_path))

        # store reverse demo transitions
        if os.path.exists(transition_path) and use_reversed_transition:
            if reversible_flags[i_env] or self_reversible_flags[i_env]:
                reverse_i_env = reversible_dict[i_env]
                reverse_env_name = env_names[reverse_i_env]
                reversal_replay_buffer = agent.reversal_replay_buffers[reverse_i_env]
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

                    if reversible_flags[i_env]:
                        reverse_target_pos = all_goal_pos[reverse_i_env]
                    if self_reversible_flags[i_env]:
                        reverse_target_pos = reverse_next_states[0][4:7]
                
                    # The trajectory is reversed, so the goal position should be changed accordingly
                    reverse_states = [change_target_pos(s, reverse_env_name, reverse_target_pos) for s in reverse_states]
                    reverse_next_states = [change_target_pos(s, reverse_env_name, reverse_target_pos) for s in reverse_next_states]

                    reverse_rewards = [state_reward(s, reverse_env_name, reward_shaping) for s in reverse_next_states]
                    # add this trajectory to reverse agent replay buffer
                    for i_sample in range(len(reverse_states)):
                        reverse_state, act, reverse_reward, reverse_next_state, done_no_max = reverse_states[i_sample], act_this_episode[i_sample], reverse_rewards[i_sample], reverse_next_states[i_sample], done_no_max_this_episode[i_sample]

                        reverse_agent_state, reverse_env_state = state2agentenv(reverse_state, reverse_env_name)
                        reverse_next_agent_state, reverse_next_env_state = state2agentenv(reverse_next_state, reverse_env_name)

                        reversal_replay_buffer.store_transition(reverse_state, reverse_agent_state, reverse_env_state, act, reverse_reward, reverse_next_state, reverse_next_agent_state, reverse_next_env_state, done_no_max)

                print('reverse env {}: load {} transitions from {} to reversal replay buffer'.format(reverse_env_name, len(transitions), transition_path))
                print('reversal reward density {:.4f}'.format(np.mean(reversal_replay_buffer.reward_memory[:reversal_replay_buffer.mem_cntr])))

    # forward and reverse reward / potential models
    if use_forward_reward:
        for i_env in range(num_envs):
            print('*'*20)
            success_replay_buffer = agent.success_replay_buffers[i_env]
            env_name = env_names[i_env]
            transition_path = 'generate/{}_transitions_{}trajectory_sparse.npy'.format(env_name, n_forward_demo)
            reward_trajectory_lengths = []
            total_env_states = []
            total_rewards = []
            total_next_env_states = []

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
                            if reward_model_type == 'potential':
                                len_trajectory = end_index + 1 - start_index
                                idx = i_step + 1 - start_index
                                if potential_type == 'linear':
                                    total_rewards.append(idx / len_trajectory)
                
                for i in range(len(total_env_states)):
                    env_state = total_env_states[i]
                    reward = total_rewards[i]
                    next_env_state = total_next_env_states[i]
                    # only env_state, reward and next_env_state are useful
                    success_replay_buffer.store_transition(state, agent_state, env_state, action, reward, next_state, next_agent_state, next_env_state, done)
                    
                print('env {}: load {} trajectories {} transitions from {} to success replay buffer'.format(env_name, len(reward_trajectory_lengths), len(total_env_states), transition_path))
            else:
                print('{} does not exist.'.format(transition_path))
        
        update_start = time.time()
        for i_epoch in range(500):
            forward_reward_loss = agent.update_potential(type='forward')
            
        print('forward_reward_loss {:.5f} update {:.2f} seconds'.format(forward_reward_loss, time.time() - update_start))
        
    if use_reversed_reward:
        for i_env in range(num_envs):
            print('*'*20)
            if reversible_flags[i_env] or self_reversible_flags[i_env]:
                reverse_replay_buffer = agent.reverse_replay_buffers[i_env]
                reverse_i_env = reversible_dict[i_env]
                env_name = env_names[i_env]
                reverse_env_name = env_names[reverse_i_env]
                reverse_transition_path = 'generate/{}_transitions_{}trajectory_sparse.npy'.format(reverse_env_name, n_reverse_demo)

                reward_trajectory_lengths = []
                total_env_states = []
                total_rewards = []
                total_next_env_states = []

                if os.path.exists(reverse_transition_path):
                    transitions = np.load(reverse_transition_path, allow_pickle=True)
                    for i_episode in range(n_reverse_demo):
                        state_this_episode = []
                        reward_this_episode = []
                        next_state_this_episode = []
                        
                        for i, transition in enumerate(transitions[i_episode*horizon:(i_episode+1)*horizon]):
                            state, action, reward, next_state, done = transition
                            state_this_episode.append(state)
                            reward_this_episode.append(reward)
                            next_state_this_episode.append(next_state)

                        # reverse this trajectory
                        reverse_states = next_state_this_episode.copy()
                        reverse_next_states = state_this_episode.copy()

                        if reversible_flags[i_env]:
                            target_pos = all_goal_pos[i_env]
                        if self_reversible_flags[i_env]:
                            target_pos = reverse_next_states[0][4:7]

                        # The trajectory is reversed, so the goal position should be changed accordingly
                        reverse_states = [change_target_pos(s, env_name, target_pos) for s in reverse_states]
                        reverse_next_states = [change_target_pos(s, env_name, target_pos) for s in reverse_next_states]
                    
                        if 1 in reward_this_episode and np.sum(reward_this_episode) > success_threshold:
                            first_reward = reward_this_episode.index(1)

                            start_index = 0
                            end_index = first_reward
                            reward_trajectory_lengths.append(end_index - start_index)
                            print(start_index, end_index)

                            for i_step in range(start_index, end_index+1):
                                state = reverse_states[i_step]
                                next_state = reverse_next_states[i_step]

                                agent_state, env_state = state2agentenv(state, env_name)
                                next_agent_state, next_env_state = state2agentenv(next_state, env_name)

                                total_env_states.append(env_state)
                                total_next_env_states.append(next_env_state)
                                if reward_model_type == 'potential':
                                    len_trajectory = end_index + 1 - start_index
                                    idx = i_step - start_index
                                    if potential_type == 'linear':
                                        total_rewards.append(1 - idx / len_trajectory)

                    for i in range(len(total_env_states)):
                        env_state = total_env_states[i]
                        reverse_reward = total_rewards[i]
                        next_env_state = total_next_env_states[i]
                        # only env_state, reward and next_env_state are useful
                        reverse_replay_buffer.store_transition(next_state, next_agent_state, next_env_state, action, reverse_reward, state, agent_state, env_state, done)
                    
                    print('env {}: load {} trajectories {} transitions from {} to reverse replay buffer'.format(env_name, len(reward_trajectory_lengths), len(total_env_states), reverse_transition_path))
        
        update_start = time.time()
        for i_epoch in range(500):
            reverse_reward_loss = agent.update_potential(type='reverse')
            
        print('reverse_reward_loss {:.5f} update {:.2f} seconds'.format(reverse_reward_loss, time.time() - update_start))

    # training of MT-SAC agent
    score_history = []
    success_history = []
    i = 0
    eval_interval = 20
    save_video_interval = eval_interval
    save_model_interval = 500
    num_eval = 20 #20
    num_sampling_episodes = 10 #10
    START = time.time()
    train_finished = np.zeros(num_envs)

    while i < num_training_episodes and (np.sum(train_finished)<num_envs):
        # the agent will interact with the environment
        if i % eval_interval == 0: #0
            #test
            print('test at episode {} train_finished {}'.format(i, train_finished))
            for i_test_env in range(num_envs):
                env_name = env_names[i_test_env]
                test_env = all_test_envs[i_test_env]
                video_dir = video_dirs[i_test_env]
                eval_df = eval_dfs[i_test_env]
                eval_df_path = eval_df_paths[i_test_env]
                env_embedding = env_embeddings[i_test_env]

                start = time.time()
                i_test = 0
                test_success_sum = 0

                while i_test < num_eval:
                    done = False
                    test_score = 0
                    env_state_this_episode = []
                    next_env_state_this_episode = []
                    obs, _ = test_env.reset()
                    action = np.zeros(action_size)
                    state = obs2state(obs, action, test_env, env_name)
                    i_step = 0
                    if save_video and i % save_video_interval == 0:
                        frames = []
                    while not done:
                        if save_video and i % save_video_interval == 0:
                            frame = test_env.render()
                            frames.append(frame)
                        i_step += 1
                        action = agent.act(state, env_embedding, sample=False)
                        try:
                            next_obs, reward, _, done, info = test_env.step(action)
                        except MujocoException as e:
                            print('got MujocoException {} at test episode {} timestep {}'.format(str(e), i_test, i_step))
                            print('state {} action {}'.format(state, action))
                            done = True
                        next_state = obs2state(next_obs, action, test_env, env_name)
                        reward = state_reward(next_state, env_name, reward_shaping)
                        test_score += reward
                        _, env_state = state2agentenv(state, env_name)
                        _, next_env_state = state2agentenv(next_state, env_name)
                        env_state_this_episode.append(env_state)
                        next_env_state_this_episode.append(next_env_state)
                        
                        obs = next_obs
                        state = next_state

                    with torch.no_grad():
                        if reward_model_type == 'potential':
                            env_embedding_tensor = torch.FloatTensor(env_embedding).repeat(len(next_env_state_this_episode), 1).to(agent.device)
                            forward_new_potential = agent.forward_potential(torch.FloatTensor(next_env_state_this_episode).to(agent.device), env_embedding_tensor)
                            forward_potential = agent.forward_potential(torch.FloatTensor(env_state_this_episode).to(agent.device), env_embedding_tensor)
                            reverse_new_potential = agent.reverse_potential(torch.FloatTensor(next_env_state_this_episode).to(agent.device), env_embedding_tensor)
                            reverse_potential = agent.reverse_potential(torch.FloatTensor(env_state_this_episode).to(agent.device), env_embedding_tensor)
                            forward_score = torch.clip((forward_new_potential - forward_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value).sum().cpu().numpy()
                            reverse_score = torch.clip((reverse_new_potential - reverse_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value).sum().cpu().numpy()

                        estimated_test_score = 0
                        if use_forward_reward: estimated_test_score += forward_score
                        if use_reversed_reward: estimated_test_score += reverse_score
                        if use_forward_reward and use_reversed_reward: estimated_test_score /= 2
                        
                    # test_success = state_reward(state, env_name, reward_shaping)
                    test_success = int(test_score > 10)
                    test_success_sum += test_success
                    print('env {} test at episode {}; i_test {}; num_steps {}; score {:.2f}; estimated score {:.2f}; success {}; time {:.2f}; total_time {:.2f}'.format(env_name, i, i_test, i_step, test_score, estimated_test_score, test_success, time.time()-start, time.time()-START))
                    eval_df.loc[len(eval_df.index)] = [i, i_test, i_step, test_score, estimated_test_score, test_success, time.time()-start, time.time()-START]
                    eval_df.to_csv(eval_df_path)

                    if save_video and i % save_video_interval == 0:
                        video_start = time.time()
                        test_video_path = os.path.join(video_dir, 'episode{}_test{}.mp4'.format(i, i_test))
                        imageio.mimsave(uri=test_video_path, ims=frames, fps=20, macro_block_size = 1)
                        print('video saved at {} in {:.3f}s'.format(test_video_path, time.time()-video_start))

                    i_test += 1

                if test_success_sum == num_eval:
                    train_finished[i_test_env] = 1
                    print('env {} train finished training'.format(env_name))
                    print('train_finished {} {}'.format(train_finished, np.sum(train_finished)))
        
        # train
        for i_env in range(num_envs):
            env_name = env_names[i_env]
            env = all_envs[i_env]
            env_embedding = env_embeddings[i_env]
            train_df = train_dfs[i_env]
            train_df_path = train_df_paths[i_env]
            train_video_dir = train_video_dirs[i_env]
            
            start = time.time()
            done = False
            score = 0
            obs, _ = env.reset()
            action = np.zeros(action_size)
            state = obs2state(obs, action, env, env_name)
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
                    frame = env.render()
                    frames.append(frame)
                i_step += 1
                if i < num_sampling_episodes:
                    # choose actions randomly
                    action = np.random.uniform(action_low, action_high)
                else:
                    # choose actions based on the actor
                    action = agent.act(state, env_embedding, sample=True)
            
                try:
                    next_obs, reward, _, done, info = env.step(action)
                except MujocoException as e:
                    print('got MujocoException {} at episode {} timestep {}'.format(str(e), i, i_step))
                    print('state {} action {}'.format(state, action))
                    done = True

                done_no_max = 0.0 if i_step == _max_episode_steps else float(done)
                next_state = obs2state(next_obs, action, env, env_name)
                reward = state_reward(next_state, env_name, reward_shaping)
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
                if i >= num_sampling_episodes:
                    train_start = time.time()
                    if (num_envs > 10 and i_env % 5 == 0) or (num_envs <= 10):
                        filter_transition = utils.filter_transition

                        critic_loss, actor_loss, alpha_loss, alpha, log_prob = agent.update(i_step, use_forward_reward, use_reversed_reward, reward_model_type, reward_model_max_value, horizon, use_reversed_transition, all_thresholds, all_state_max, all_state_min, filter_transition, filter_type)                        

                        for_loss, inv_loss = 0, 0
                        forward_reward_loss, reverse_reward_loss, reversible_ratio = 0, 0, 0
                        if i_step % num_envs == 0 and use_reversed_transition:
                            for_loss = agent.update_forward_dynamics()
                            inv_loss = agent.update_inverse_dynamics()
                    else:
                        critic_loss, actor_loss, alpha_loss, alpha, log_prob, for_loss, inv_loss, forward_reward_loss, reverse_reward_loss, reversible_ratio = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                        
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

            score_history.append(score)
            success = state_reward(state, env_name, reward_shaping)
            success_history.append(success)

            # store transitions of this episode to the replay buffer
            replay_buffer = agent.replay_buffers[i_env]
            for i_sample in range(len(state_this_episode)):
                state, agent_state, env_state, act, reward, next_state, next_agent_state, next_env_state, done_no_max = state_this_episode[i_sample], agent_state_this_episode[i_sample], env_state_this_episode[i_sample], act_this_episode[i_sample], reward_this_episode[i_sample], next_state_this_episode[i_sample], next_agent_state_this_episode[i_sample], next_env_state_this_episode[i_sample], done_no_max_this_episode[i_sample]
                replay_buffer.store_transition(state, agent_state, env_state, act, reward, next_state, next_agent_state, next_env_state, done_no_max)

            reverse_rewards = [0]
            if use_reversed_transition and (reversible_flags[i_env] or self_reversible_flags[i_env]):
                reverse_i_env = reversible_dict[i_env]
                reverse_env_name = env_names[reverse_i_env]
                reversal_replay_buffer = agent.reversal_replay_buffers[reverse_i_env]
                # reverse this trajectory
                reverse_states = next_state_this_episode.copy()
                reverse_next_states = state_this_episode.copy()
                
                # The trajectory is reversed, so the goal position should be changed accordingly
                if reversible_flags[i_env]:
                    reverse_target_pos = all_goal_pos[reverse_i_env]
                if self_reversible_flags[i_env]:
                    reverse_target_pos = reverse_next_states[0][4:7]
                reverse_states = [change_target_pos(s, reverse_env_name, reverse_target_pos) for s in reverse_states]
                reverse_next_states = [change_target_pos(s, reverse_env_name, reverse_target_pos) for s in reverse_next_states]

                reverse_rewards = [state_reward(s, reverse_env_name, reward_shaping) for s in reverse_next_states]

                # add high value trajectories to reverse agent replay buffer
                if np.sum(reverse_rewards) > 0:
                    for i_sample in range(len(reverse_states)):
                        reverse_state, act, reverse_reward, reverse_next_state, done_no_max = reverse_states[i_sample], act_this_episode[i_sample], reverse_rewards[i_sample], reverse_next_states[i_sample], done_no_max_this_episode[i_sample]

                        reverse_agent_state, reverse_env_state = state2agentenv(reverse_state, reverse_env_name)
                        reverse_next_agent_state, reverse_next_env_state = state2agentenv(reverse_next_state, reverse_env_name)

                        reversal_replay_buffer.store_transition(reverse_state, reverse_agent_state, reverse_env_state, act, reverse_reward, reverse_next_state, reverse_next_agent_state, reverse_next_env_state, done_no_max)
            
            with torch.no_grad():
                if reward_model_type == 'potential':
                    env_embedding_tensor = torch.FloatTensor(env_embedding).repeat(len(next_env_state_this_episode), 1).to(agent.device)
                    forward_new_potential = agent.forward_potential(torch.FloatTensor(next_env_state_this_episode).to(agent.device), env_embedding_tensor)
                    forward_potential = agent.forward_potential(torch.FloatTensor(env_state_this_episode).to(agent.device), env_embedding_tensor)
                    reverse_new_potential = agent.reverse_potential(torch.FloatTensor(next_env_state_this_episode).to(agent.device), env_embedding_tensor)
                    reverse_potential = agent.reverse_potential(torch.FloatTensor(env_state_this_episode).to(agent.device), env_embedding_tensor)
                    forward_score = torch.clip((forward_new_potential - forward_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value).sum().cpu().numpy()
                    reverse_score = torch.clip((reverse_new_potential - reverse_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value).sum().cpu().numpy()

                estimated_score = 0
                if use_forward_reward: estimated_score += forward_score
                if use_reversed_reward: estimated_score += reverse_score
                if use_forward_reward and use_reversed_reward: estimated_score /= 2

            if use_forward_reward:
                success_replay_buffer = agent.success_replay_buffers[i_env]
                if success == 1 and np.sum(reward_this_episode) > success_threshold:                    
                    first_reward = reward_this_episode.index(1)
                
                    start_index = 0
                    end_index = first_reward
                    print(start_index, end_index)

                    for i_env_step in range(start_index, end_index+1):
                        env_state = env_state_this_episode[i_env_step]
                        next_env_state = next_env_state_this_episode[i_env_step]
                        if reward_model_type == 'potential':
                            len_trajectory = end_index + 1 - start_index
                            idx = i_env_step + 1 - start_index
                            if potential_type == 'linear':
                                reward = idx / len_trajectory
                        # only env_state, reward and next_env_state are useful
                        success_replay_buffer.store_transition(state, agent_state, env_state, action, reward, next_state, agent_state, next_env_state, done)
                    
            if use_reversed_reward and (reversible_flags[i_env] or self_reversible_flags[i_env]):
                reverse_i_env = reversible_dict[i_env]
                reverse_env_name = env_names[reverse_i_env]
                reverse_replay_buffer = agent.reverse_replay_buffers[reverse_i_env]
                if success == 1 and np.sum(reward_this_episode) > success_threshold:
                    first_reward = reward_this_episode.index(1)
                    start_index = 0
                    end_index = first_reward
                    print(start_index, end_index)

                    for i_env_step in range(start_index, end_index+1):
                        state = state_this_episode[i_env_step]
                        next_state = next_state_this_episode[i_env_step]

                        # The trajectory is reversed, so the goal position should be changed accordingly.
                        if reversible_flags[i_env]:
                            reverse_target_pos = all_goal_pos[reverse_i_env]
                        if self_reversible_flags[i_env]:
                            reverse_target_pos = reverse_next_states[0][4:7]
                        state = change_target_pos(state, reverse_env_name, reverse_target_pos)
                        next_state = change_target_pos(next_state, reverse_env_name, reverse_target_pos)

                        agent_state, env_state = state2agentenv(state, reverse_env_name)
                        next_agent_state, next_env_state = state2agentenv(next_state, reverse_env_name)

                        if reward_model_type == 'potential':
                            len_trajectory = end_index + 1 - start_index
                            idx = i_env_step - start_index
                            if potential_type == 'linear':
                                reverse_reward = 1 - idx / len_trajectory
                        reverse_replay_buffer.store_transition(next_state, agent_state, next_env_state, action, reverse_reward, state, agent_state, env_state, done)
                    
            replay_buffer = agent.replay_buffers[i_env]
            demo_replay_buffer = agent.demo_replay_buffers[i_env]
            success_replay_buffer = agent.success_replay_buffers[i_env]
            reverse_replay_buffer = agent.reverse_replay_buffers[i_env]
            reversal_replay_buffer = agent.reversal_replay_buffers[i_env]

            print('env {}; train at episode {}; num_steps {}; train score {:.2f}; 100game avg score {:.2f}; estimated_score {:.3f}; success {}; 100 game avg success {:.2f}; critic_loss {:.5f}; actor_loss {:.5f}; forward_reward_loss {:.5f}; reverse_reward_loss {:.5f}; reversible_ratio {:.5f} for_loss {:.5f}; inv_loss {:.5f}; alpha_loss {:.5f}; alpha {:.5f}; log_prob {:.5f}; env_buffer {}; demo_buffer {}; success_buffer {}; reverse_buffer {}; reversal_buffer {}; train_time {:.3f}; episode_time {:.2f}; total_time {:.2f}'.format(env_name, i, i_step, score, np.mean(score_history[-100:]), estimated_score, success, np.mean(success_history[-100:]), np.mean(critic_losses), np.mean(actor_losses), np.mean(forward_reward_losses), np.mean(reverse_reward_losses), np.mean(reversible_ratios), np.mean(for_losses), np.mean(inv_losses), np.mean(alpha_losses), alpha, np.mean(log_probs), replay_buffer.mem_cntr, demo_replay_buffer.mem_cntr, success_replay_buffer.mem_cntr, reverse_replay_buffer.mem_cntr, reversal_replay_buffer.mem_cntr, train_time, time.time()-start, time.time()-START))

            train_df.loc[len(train_df.index)] = [i, i_step, score, success, estimated_score, np.mean(critic_losses), np.mean(actor_losses), np.mean(forward_reward_losses), np.mean(reverse_reward_losses), np.mean(reversible_ratios), np.mean(for_losses), np.mean(inv_losses), np.mean(alpha_losses), alpha, np.mean(log_probs), replay_buffer.mem_cntr, demo_replay_buffer.mem_cntr, success_replay_buffer.mem_cntr, reverse_replay_buffer.mem_cntr, reversal_replay_buffer.mem_cntr, train_time, time.time()-start, time.time()-START]
            train_df.to_csv(train_df_path)

            if save_video_train:
                video_start = time.time()
                train_video_path = os.path.join(train_video_dir, 'train_episode{}.mp4'.format(i))
                imageio.mimsave(uri=train_video_path, ims=frames, fps=20, macro_block_size = 1)
                print('video saved at {} in {:.3f}s'.format(train_video_path, time.time()-video_start))

        if i % 10 == 0:
            for i_epoch in range(50):
                if reward_model_type == 'potential':
                    forward_reward_loss = agent.update_potential(type='forward')
                    reverse_reward_loss = agent.update_potential(type='reverse')        

        i += 1