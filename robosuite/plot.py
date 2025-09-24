import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import warnings 
warnings.filterwarnings('ignore')

def run_average(values, run_average_episode):
    values_copy = values.copy()
    
    return np.array([np.mean(values_copy[max(0, i-run_average_episode+1):i+1]) for i in range(len(values_copy))])

def read_train_fname(train_fname, run_average_episode=100, end_episode=500):
    train_episodes, train_rewards, train_success = [], [], []
    if os.path.exists(train_fname):
        train_df = pd.read_csv(train_fname)
        train_episodes = np.array(train_df['episode'].values)[:end_episode]
        train_rewards = np.array(train_df['score'].values)[:end_episode]
        train_success = np.array(train_df['success'].values)[:end_episode]
        
        train_rewards = run_average(train_rewards, run_average_episode)
        train_success = run_average(train_success, run_average_episode)
        
    try:
        num_points = list(train_episodes).index(end_episode)
    except:
        num_points = len(train_episodes)

    return train_episodes[:num_points], train_rewards[:num_points], train_success[:num_points]

def read_eval_fname(eval_fname, run_average_episode=5, end_episode=500, success_threshold=0.9, eval_interval=20):
    eval_episodes, eval_rewards, eval_success = [], [], []
    if os.path.exists(eval_fname):
        eval_df = pd.read_csv(eval_fname)
        eval_rewards_group = eval_df.groupby('episode').apply(lambda x:x['eval_score'].mean())
        eval_success_group = eval_df.groupby('episode').apply(lambda x:x['eval_success'].mean())
        eval_episodes = np.array(eval_rewards_group.index)[:end_episode//eval_interval]
        eval_episodes = np.arange(end_episode // eval_interval) * eval_interval
        eval_rewards = np.array(eval_rewards_group.values)[:end_episode//eval_interval]
        eval_success = np.array(eval_success_group.values)[:end_episode//eval_interval]
        
        success_episodes = []
        eval_success_max = -1
        for i, success in enumerate(eval_success):
            if success >= success_threshold:
                success_episodes.append(i)
            if eval_success[i] > eval_success_max: eval_success_max = max(eval_success[i], eval_success_max)
            eval_success[i] = max(eval_success_max, eval_success[i])
        
        if len(eval_rewards) < end_episode // eval_interval:
            eval_rewards = np.concatenate([eval_rewards, np.repeat(eval_rewards[-1], end_episode // eval_interval-len(eval_rewards))])
            eval_success = np.concatenate([eval_success, np.repeat(eval_success[-1], end_episode // eval_interval-len(eval_success))])
        eval_rewards = run_average(eval_rewards, run_average_episode)
        eval_success = run_average(eval_success, run_average_episode)
        
    num_points = len(eval_episodes)
   
    return eval_episodes[:num_points], eval_rewards[:num_points], eval_success[:num_points]

def compute_mean_std(data):
    # return the mean and std of data
    # data can be sequences with different lengths

    data_length = [len(d) for d in data]
    mean = []
    std = []
    for i in range(max(data_length)):
        tmp = []
        for j in range(len(data)):
            if i < data_length[j]:
                tmp.append(data[j][i])
        mean.append(np.mean(tmp))
        std.append(np.std(tmp))
    
    return np.array(mean), np.array(std)

if __name__ == '__main__':
    env_names = ['Door', 'Door_Close']
    # env_names = ['Old_Door', 'Old_Door_Close']
    # env_names = ['Stack', 'UnStack']
    # env_names = ['TwoArmPegInHole', 'TwoArmPegRemoval']
    # env_names = ['NutAssemblyRound', 'NutDisAssemblyRound']
    
    #####
    # algo names
    # algos = ['SAC_agentenv_2agents_10demo_sparse', 'SAC_agentenv_2agents_10demo_use_reversed_transition_state_max_diff0.01_sparse', 'SAC_agentenv_2agents_10demo_use_reversed_reward_10reverse_demo_use_forward_reward_10demo_separate_potential_model_linear_potential_sparse', 'SAC_agentenv_2agents_10demo_use_reversed_reward_10reverse_demo_use_forward_reward_10demo_separate_potential_model_linear_potential_use_reversed_transition_state_max_diff0.01_sparse']
    algos = ['SAC_agentenv_2agents_10demo_sparse', 'SAC_agentenv_2agents_10demo_use_reversed_transition_state_max_diff0.01_sparse', 'SAC_agentenv_2agents_10demo_use_reversed_reward_10reverse_demo_use_forward_reward_10demo_separate_potential_model_linear_potential_sparse', 'single_task_tr_sac']

    algo_name_dict = {}
    algo_name_dict['SAC_agentenv_2agents_10demo_sparse'] = 'Single-Task SAC'
    algo_name_dict['SAC_agentenv_2agents_10demo_use_reversed_transition_state_max_diff0.01_sparse'] = '+reversal aug'
    algo_name_dict['SAC_agentenv_2agents_10demo_use_reversed_reward_10reverse_demo_use_forward_reward_10demo_separate_potential_model_linear_potential_sparse'] = '+reversal reward shaping'
    algo_name_dict['SAC_agentenv_2agents_10demo_use_reversed_reward_10reverse_demo_use_forward_reward_10demo_separate_potential_model_linear_potential_use_reversed_transition_state_max_diff0.01_sparse'] = 'Single-Task TR-SAC'
    algo_name_dict['single_task_tr_sac'] = 'Single-Task TR-SAC'
    #####

    #####
    # result directory
    # Assume that results for each env_name and algo have been moved to `results/env_name/algo`. 
    # For example, results for Door Opening Outward and SAC have been moved to `results/Door/SAC_10demo_sparse`.
    result_directory = "results/"
    #####

    seeds = np.arange(1, 6)
    
    colors = ['black', 'blue', 'green', 'red', 'purple', 'brown', 'yellow', 'gray', 'orange', 'deeppink']

    plot_train = not True
    plot_eval = True
    end_episode = 500
    run_average_episode = 100

    fig = plt.figure(figsize=(5,10))
    num_row = len(env_names)
    num_col = 1
    eval_linestyle = 'solid'
    
    plot_env_name_dict = {}
    plot_env_name_dict['Door'] = 'Door Open Outward'
    plot_env_name_dict['Door_Close'] = 'Door Close Outward'
    plot_env_name_dict['Old_Door'] = 'Door Open Inward'
    plot_env_name_dict['Old_Door_Close'] = 'Door Close Inward'
    plot_env_name_dict['TwoArmPegInHole'] = 'Peg Insertion'
    plot_env_name_dict['TwoArmPegRemoval'] = 'Peg Removal'
    plot_env_name_dict['NutAssemblyRound'] = 'Nut Assembly'
    plot_env_name_dict['NutDisAssemblyRound'] = 'Nut Disassembly'
    plot_env_name_dict['Stack'] = 'Block Stack'
    plot_env_name_dict['UnStack'] = 'Block Unstack'   

    for i_env, env_name in enumerate(env_names):
        plot_env_name = plot_env_name_dict[env_name]
        for i_algo, algo in enumerate(algos):
            print('-'*30)
            algo_name = algo_name_dict[algo]
            print(env_name, algo_name)
            algo_color = colors[i_algo]
            linestyle = 'solid'
            directory = os.path.join(result_directory, env_name, algo)
            
            if plot_eval:
                eval_fnames = [os.path.join(directory, '{}_seed{}'.format(env_name, seed), 'eval.csv') for seed in seeds]
                total_eval_rewards = []
                total_eval_success = []
                total_eval_episodes = []
                for eval_fname in eval_fnames:
                    eval_episodes, eval_rewards, eval_success = read_eval_fname(eval_fname, run_average_episode = run_average_episode // 20, end_episode = end_episode)
                    
                    if len(eval_episodes) > len(total_eval_episodes): total_eval_episodes = eval_episodes
                    total_eval_rewards.append(eval_rewards)
                    total_eval_success.append(eval_success)
                eval_rewards_mean, eval_rewards_std = compute_mean_std(total_eval_rewards)
                eval_success_mean, eval_success_std = compute_mean_std(total_eval_success)

                for _ in range(len(total_eval_episodes)):
                    if total_eval_episodes[_] in [100, 200, 300, 400, 480]:
                        print('episode {} eval success {:.2f}$\pm${:.2f}'.format(total_eval_episodes[_], eval_success_mean[_], eval_success_std[_]))

                plt.subplot(num_row * 100 + num_col * 10 + (i_env*num_col + 1))
                plt.plot(total_eval_episodes, eval_success_mean, color=algo_color, label='{}'.format(algo_name), linestyle=linestyle)
                plt.fill_between(total_eval_episodes, eval_success_mean - eval_success_std, eval_success_mean + eval_success_std, alpha=0.2, color=algo_color)
        
                if num_col > 1:
                    plt.subplot(num_row * 100 + num_col * 10 + (i_env*num_col + 1) + 1)
                    plt.plot(total_eval_episodes, eval_rewards_mean, color=algo_color, label='{}'.format(algo_name), linestyle=linestyle)
                    plt.fill_between(total_eval_episodes, eval_rewards_mean-eval_rewards_std, eval_rewards_mean+eval_rewards_std, alpha=0.2, color=algo_color)
                
        for _ in range(num_col):
            plt.subplot(num_row * 100 + num_col * 10 + (i_env*num_col + 1) + _)
            plt.grid()
            plt.legend(loc='best')
            plt.xlabel('episodes')

            if _ == 0:
                plt.ylabel('success')
                plt.xlim(0, 500)
                plt.ylim(-0.1, 1 * 1.1)
                # plt.title('success {} moving average {}'.format(plot_env_name, run_average_episode))    
                plt.title('{}'.format(plot_env_name))    
            if _ == 1:
                plt.ylabel('rewards')
                plt.ylim(-50, 500 * 1.1)
                # plt.title('rewards {} moving average {}'.format(plot_env_name, run_average_episode))
                plt.title('rewards {}'.format(plot_env_name))
            
    plt.subplots_adjust(hspace = 0.3)
    plt.savefig('plots/{}_{}'.format(env_names[0], env_names[1]))
    plt.close()
    # plt.show()