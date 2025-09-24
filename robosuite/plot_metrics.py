import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import time

def run_average(values, run_average_episode):
    values_copy = values.copy()

    return np.array([np.mean(values_copy[max(0, i-run_average_episode+1):i+1]) for i in range(len(values_copy))])

def read_eval_fname(eval_fname, run_average_episode=1, end_episode=500, success_threshold=0.9, eval_interval=20):
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
            eval_rewards = np.concatenate([eval_rewards, np.repeat(eval_rewards[-1], 25-len(eval_rewards))])
            eval_success = np.concatenate([eval_success, np.repeat(eval_success[-1], 25-len(eval_success))])
        eval_rewards = run_average(eval_rewards, run_average_episode)
        eval_success = run_average(eval_success, run_average_episode)

    num_points = len(eval_episodes)
   
    return eval_episodes[:num_points], eval_rewards[:num_points], eval_success[:num_points]

if __name__ == '__main__':
    n_demo = 10
        
    linestyle_dict = {}
    algo_color_dict = {}
    colors = ['black', 'blue', 'green', 'red', 'purple', 'brown', 'yellow', 'gray', 'orange', 'deeppink']
    
    end_episode = 500
    run_average_episode = 20

    seed_env_success_dict = {}
    algo_names = []
    algo_name_dict = {}
    seeds = np.arange(1, 6)
    env_names = ['Door', 'Door_Close', 'Old_Door', 'Old_Door_Close', 'TwoArmPegInHole', 'TwoArmPegRemoval', 'NutAssemblyRound', 'NutDisAssemblyRound', 'Stack', 'UnStack']
    result_directory = 'results'

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

    for i_env, env_name in enumerate(env_names):
        print('env_name {}'.format(env_name))
        for i_algo, algo in enumerate(algos):
            algo_name = algo_name_dict[algo]
            eval_csv_exist = False
            directory = os.path.join(result_directory, env_name, algo)

            for seed in seeds:
                eval_fname = os.path.join(directory, '{}_seed{}'.format(env_name, seed), 'eval.csv')
                if os.path.exists(eval_fname):
                    eval_csv_exist = True
                    eval_episodes, eval_rewards, eval_success = read_eval_fname(eval_fname, run_average_episode = run_average_episode // 20, end_episode = end_episode)
                    if len(eval_success) < 25:
                        # fill results for early stop with the last value
                        eval_success = np.concatenate([eval_success, np.repeat(eval_success[-1], 25-len(eval_success))])
                    seed_env_success_dict[(algo_name, seed, '{}{}demo'.format(env_name, n_demo))] = eval_success
            
            if algo_name not in algo_names and eval_csv_exist: algo_names.append(algo_name)

    all_success_dict = {}
    for algo_name in algo_names:
        all_success_dict[algo_name] = []
        for seed in seeds:
            seed_success = []
            for env_name in env_names:
                if (algo_name, seed, '{}{}demo'.format(env_name, n_demo)) in seed_env_success_dict:
                    seed_success.append(seed_env_success_dict[(algo_name, seed, '{}{}demo'.format(env_name, n_demo))])
            all_success_dict[algo_name].append(seed_success)
        all_success_dict[algo_name] = np.array(all_success_dict[algo_name])
    
    eval_episodes = np.arange(25)
    ale_episode_success_dict = {algo: success[:, :, eval_episodes] for algo, success in all_success_dict.items()}
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., episode]) for episode in range(scores.shape[-1])])
    
    start = time.time()
    print('Computing IQM...')
    iqm_success, iqm_cis = rly.get_interval_estimates(ale_episode_success_dict, iqm)
    print('compute iqm time: {:.2f} seconds'.format(time.time()-start))

    iqm_df = pd.DataFrame(columns = eval_episodes)
    iqm_df.insert(0, 'iqm', [])
    iqm_df.insert(0, 'algo_name', [])
    figsize = (5, 5)
    plt.figure(figsize=figsize)
    for i_algo, algo_name in enumerate(algo_names):
        metric_values = iqm_success[algo_name]
        lower, upper = iqm_cis[algo_name]
        
        iqm_df.loc[len(iqm_df.index)] = np.concatenate([[algo_name,'iqm_success'], metric_values])
        iqm_df.loc[len(iqm_df.index)] = np.concatenate([[algo_name,'iqm_lower'], lower])
        iqm_df.loc[len(iqm_df.index)] = np.concatenate([[algo_name,'iqm_upper'], upper])
        
        if algo_name in algo_color_dict:
            color = algo_color_dict[algo_name]
        else:
            color = colors[i_algo]
        if algo_name in linestyle_dict:
            linestyle = linestyle_dict[algo_name]
        else:
            linestyle = 'solid'

        plt.plot(eval_episodes, metric_values, color=color, linestyle=linestyle, label=algo_name)
        plt.fill_between(eval_episodes, y1=lower, y2=upper, color=color, alpha=0.2)

    print(iqm_df)
    iqm_df.to_csv('plots/IQM_robosuite_df.csv')

    plt.xlabel('Number of Episodes (x20)')
    plt.ylabel('IQM Eval Success')
    plt.title('IQM Eval Success')
    plt.grid()
    plt.legend(loc='best')
    # plt.show()
    plt.savefig('plots/IQM_robosuite')
    plt.close()