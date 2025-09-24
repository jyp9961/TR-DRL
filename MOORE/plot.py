import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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
    task_name = 'MT50'
    plot_dir = 'plots/{}'.format(task_name)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    algo_names = ['moore', 'trmoore']
    
    colors = ['black', 'blue', 'green', 'red', 'purple', 'brown', 'yellow', 'gray', 'orange', 'deeppink']
    if task_name == 'MT10':
        seeds = np.arange(1, 11)
        num_envs = 10
        plot_type = 'MT10'
        env_names = ['reach-v2', 'push-v2', 'pick-place-v2', 'door-open-v2', 'drawer-open-v2', 'drawer-close-v2', 'button-press-topdown-v2', 'peg-insert-side-v2', 'window-open-v2', 'window-close-v2']
    if task_name == 'MT50':
        seeds = np.arange(1, 6)
        num_envs = 50
        plot_type = 'MT50'
        env_names = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-close-v2', 'faucet-open-v2', 'hammer-v2', 'hand-insert-v2','handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'peg-unplug-side-v2', 'pick-out-of-hole-v2', 'pick-place-v2', 'pick-place-wall-v2', 'plate-slide-back-side-v2', 'plate-slide-back-v2', 'plate-slide-side-v2', 'plate-slide-v2', 'push-back-v2', 'push-v2', 'push-wall-v2', 'reach-v2', 'reach-wall-v2', 'shelf-place-v2', 'soccer-v2', 'stick-pull-v2','stick-push-v2', 'sweep-into-v2', 'sweep-v2', 'window-close-v2', 'window-open-v2']

    #'''
    for i_env, env_name in enumerate(env_names):
        for i_algo, algo_name in enumerate(algo_names):
            env_success_rate = []
            env_steps = []
            for i_seed, seed in enumerate(seeds):
                result_fname = 'results/{}/{}/seed_{}/result.csv'.format(task_name, algo_name, seed)
                if os.path.exists(result_fname):
                    result_df = pd.read_csv(result_fname)
                else:
                    print('{} does not exist.'.format(result_fname))
                    continue
                success_rate = result_df[result_df['env_name']==env_name].groupby('step').apply(lambda x:np.mean(x['success_rate'])).values
                steps = np.unique(result_df['step'].values)
                
                if algo_name == 'trmoore':
                    success_rate = success_rate[::10]
                    steps = steps[::10]

                env_success_rate.append(success_rate)
                if len(steps) > len(env_steps): env_steps = steps
            if len(env_success_rate) > 0:
                env_success_rate_mean, env_success_rate_std = compute_mean_std(env_success_rate)
            else:
                continue

            color = colors[i_algo]
            plt.plot(env_steps, env_success_rate_mean, color=color, label=algo_name)
            plt.fill_between(env_steps, env_success_rate_mean-env_success_rate_std, env_success_rate_mean+env_success_rate_std, color=color, alpha=0.2)
        
            print('-'*30)
            print(algo_name, env_name)
            # print(env_success_rate)
            for i_step, step in enumerate(env_steps):
                print('step {:,} success_rate {:.1f}$\pm${:.1f}'.format(step, env_success_rate_mean[i_step]*100, env_success_rate_std[i_step]*100))
            
        plt.xlabel('Step')
        plt.ylabel('Success Rate')
        plt.ylim(-0.1, 1.1)
        plt.grid()
        plt.title('{}'.format(env_name))
        plt.legend(loc='best')
        plt.savefig(os.path.join(plot_dir, env_name))
        plt.close()
        # plt.show()
    #'''

    for i_algo, algo_name in enumerate(algo_names):
        all_success_rate = []
        all_steps = []
        for i_seed, seed in enumerate(seeds):
            result_fname = 'results/{}/{}/seed_{}/result.csv'.format(task_name, algo_name, seed)
            if os.path.exists(result_fname):
                result_df = pd.read_csv(result_fname)
            else:
                continue

            success_rate = result_df.groupby('step').apply(lambda x:np.mean(x['success_rate'])).values
            # success_rate = result_df[result_df['env_name'].isin(env_names)].groupby('step').apply(lambda x:np.mean(x['success_rate'])).values
            # success_rate = result_df[~result_df['env_name'].isin(env_names)].groupby('step').apply(lambda x:np.mean(x['success_rate'])).values
            steps = np.unique(result_df['step'].values)
            
            all_success_rate.append(success_rate)
            if len(steps) > len(all_steps): all_steps = steps
            # print('seed {}: success_rate {}'.format(seed, success_rate))
        
        if len(all_success_rate) > 0:
            success_rate_mean, success_rate_std = compute_mean_std(all_success_rate)
        else:
            continue
        
        color = colors[i_algo]
        plt.plot(all_steps * num_envs, success_rate_mean, color=color, label=algo_name)
        plt.fill_between(all_steps * num_envs, success_rate_mean-success_rate_std, success_rate_mean+success_rate_std, color=color, alpha=0.2)
    
        print('-'*30)
        print(algo_name)
        for i_step, step in enumerate(all_steps):
            print('step {:,} env step {:,} success_rate {:.1f}$\pm${:.1f}'.format(step*num_envs, step, success_rate_mean[i_step]*100, success_rate_std[i_step]*100))

    plt.xlabel('Total Env Steps')
    plt.ylabel('Success Rate')
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.title(plot_type)
    plt.legend(loc='best')
    plt.savefig(os.path.join(plot_dir, plot_type))
    plt.close()        
    # plt.show()