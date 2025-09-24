import numpy as np
# deeplearning frameworks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# framework
from . import SAC, SACPolicy
from moore.utils.replay_memory import ReplayMemory
import utils
from utils import reverse_transition, state_reward

class TRMTSACPolicy(SACPolicy):

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        idx, state = state[0], state[1]
        # modified for sharing mu and sigma
        if self._shared_mu_sigma:
            a = self._approximator.predict(state, c=idx, output_tensor=True)
            if a.ndim == 1:
                mu, log_sigma = a[:a.shape[-1]//2], a[a.shape[-1]//2:]
            else:
                mu, log_sigma = a[:, :a.shape[-1]//2], a[:, a.shape[-1]//2:]
        else:
            mu = self._mu_approximator.predict(state, c=idx, output_tensor=True)
            log_sigma = self._sigma_approximator.predict(state, c=idx, output_tensor=True)
        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())

        return torch.distributions.Normal(mu, log_sigma.exp())

class ForwardDynamics(nn.Module):
    """Forward Dynamics network"""
    def __init__(self, state_size, action_size, hidden_dim, hidden_depth, device):
        super().__init__()
        self.input_dims = state_size + action_size
        self.output_dims = state_size
        self.device = device

        self.for_dynamics = utils.mlp(self.input_dims, hidden_dim, self.output_dims, hidden_depth, output_mod = nn.Tanh()).to(self.device)
        self.apply(utils.weight_init)
        self.optimizer = torch.optim.Adam(self.for_dynamics.parameters(), lr=3e-4)

    def forward(self, state, action):
        if isinstance(state, np.ndarray): state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray): action = torch.FloatTensor(action)
        assert state.size(0) == action.size(0)
        state = state.to(self.device)
        action = action.to(self.device)

        state_actions = torch.cat([state, action], dim=-1)
        # the forward dynamics model predicts the difference between states and new_states
        diff_state = self.for_dynamics(state_actions)
        new_state = state + diff_state

        return new_state

    def update(self, state, action, next_state):
        if isinstance(next_state, np.ndarray): next_state = torch.FloatTensor(next_state)
        next_state = next_state.to(self.device)
        
        predicted_next_state = self.forward(state, action)
        loss = F.mse_loss(next_state, predicted_next_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return utils.to_np(loss)

class InverseDynamics(nn.Module):
    """Inverse Dynamics network"""
    def __init__(self, state_size, action_size, hidden_dim, hidden_depth, device):
        super().__init__()
        self.input_dims = state_size * 2
        self.output_dims = action_size
        self.device = device
        
        self.inv_dynamics = utils.mlp(self.input_dims, hidden_dim, self.output_dims, hidden_depth, output_mod = nn.Tanh()).to(self.device)
        self.apply(utils.weight_init)
        self.optimizer = torch.optim.Adam(self.inv_dynamics.parameters(), lr=3e-4)

    def forward(self, state, next_state):
        if isinstance(state, np.ndarray): state = torch.FloatTensor(state)
        if isinstance(next_state, np.ndarray): next_state = torch.FloatTensor(next_state)
        assert state.size(0) == next_state.size(0)
        state = state.to(self.device)
        next_state = next_state.to(self.device)

        states = torch.cat([state, next_state-state], dim=-1)
        actions = self.inv_dynamics(states)

        return actions

    def update(self, state, action, next_state):
        if isinstance(action, np.ndarray): action = torch.FloatTensor(action)
        action = action.to(self.device)

        predicted_action = self.forward(state, next_state)
        loss = F.mse_loss(action, predicted_action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return utils.to_np(loss)

class TRMTSAC(SAC):
    def __init__(self, env_names, reversible_indices, reversible_dict, dynamics_to_env_dict, env_to_dynamics_dict, 
                mdp_info, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                actor_params=None, actor_mu_params=None, actor_sigma_params=None, 
                log_std_min=-20, log_std_max=2, target_entropy=None, log_alpha = None, 
                critic_fit_params=None, shared_mu_sigma = False, n_contexts = 1):
        
        super().__init__(mdp_info, actor_optimizer, critic_params, batch_size,
                        initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                        actor_params=actor_params, actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                        log_std_min=log_std_min, log_std_max=log_std_max, target_entropy=target_entropy, critic_fit_params=critic_fit_params, 
                        policy_class=TRMTSACPolicy, shared_mu_sigma=shared_mu_sigma)
        
        self.env_names = env_names
        self._n_contexts = n_contexts
        
        self._replay_memory = [ReplayMemory(initial_replay_size, max_replay_size) for _ in range(n_contexts)]
        
        if log_alpha is None:
            self._log_alpha = torch.tensor([0.]*n_contexts, dtype=torch.float32)
        else:
            assert len(log_alpha) == n_contexts
            self._log_alpha = torch.tensor(log_alpha, dtype=torch.float32)

        if self.policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)
        
        self._add_save_attr(_n_contexts='primitive')

        self.reversible_indices = reversible_indices
        self.reversible_dict = reversible_dict
        self.dynamics_to_env_dict = dynamics_to_env_dict
        self.env_to_dynamics_dict = env_to_dynamics_dict
        self._reversal_replay_memory = [ReplayMemory(initial_replay_size, max_replay_size) for _ in self.reversible_indices]        

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        step_state_size, action_size = 18, 4
        hidden_dim, hidden_depth = 400, 2
        self.forward_dynamics = [ForwardDynamics(step_state_size, action_size, hidden_dim, hidden_depth, self.device) for _ in range(len(self.dynamics_to_env_dict))]
        self.inverse_dynamics = [InverseDynamics(step_state_size, action_size, hidden_dim, hidden_depth, self.device) for _ in range(len(self.dynamics_to_env_dict))]

        self.reversible_ratio = []

    def fit(self, dataset, **info):
        contexts = np.array([d[0][0] for d in dataset]).ravel().astype(np.int64)
        unique_contexts = np.unique(contexts) #np.arange(num_envs)
        for c in unique_contexts:
            idxs = np.argwhere(contexts == c).ravel()
            d = [dataset[idx] for idx in idxs]
            self._replay_memory[c].add(d)

            if c in self.reversible_indices:
                # reverse (state, next_state) in d, relabel the goal according to reverse env
                reverse_d = [reverse_transition(d_, self.reversible_dict, self.env_names) for d_ in d]
                
                # store this reversed transition in reversal replay memory
                reverse_c = self.reversible_dict[c]
                reverse_replay_buffer_id = self.reversible_indices.index(reverse_c)
                self._reversal_replay_memory[reverse_replay_buffer_id].add(reverse_d)

        fit_condition = np.all([rm.initialized for rm in self._replay_memory])

        if fit_condition:
            state_idx = []
            state = []
            action = []
            reward = []
            next_state = []
            absorbing = []

            for i in range(len(self._replay_memory)):
                state_i, action_i, reward_i, next_state_i,\
                    absorbing_i, _ = self._replay_memory[i].get(
                        self._batch_size())
                state_idx.append(np.ones(self._batch_size(), dtype=np.int64) * i)
                state.append(state_i)
                action.append(action_i)
                reward.append(reward_i)
                next_state.append(next_state_i)
                absorbing.append(absorbing_i)
                
            state_idx = np.vstack(state_idx).reshape(-1)
            state = np.vstack(state)
            action = np.vstack(action)
            reward = np.hstack(reward)
            next_state = np.vstack(next_state)
            absorbing = np.hstack(absorbing)

            # print('-'*10, 'before aug')
            # print('np.shape(state_idx)',np.shape(state_idx))
            # print('np.shape(state)',np.shape(state))
            # print('np.shape(action)', np.shape(action))
            # print('np.shape(reward)', np.shape(reward))
            # print('np.shape(next_state)', np.shape(next_state))
            # print('np.shape(absorbing)', np.shape(absorbing))

            # update forward and inverse dynamics
            if self._replay_memory[0].size > self._warmup_transitions():
                for i_dyn in self.dynamics_to_env_dict:
                    env_indices = self.dynamics_to_env_dict[i_dyn]
                    dyn_state = []
                    dyn_action = []
                    dyn_next_state = []
                    for i_c, c in enumerate(env_indices):
                        dyn_state.append(state[:, 0:18])
                        dyn_action.append(action)
                        dyn_next_state.append(next_state[:, 0:18])
                    dyn_state = np.vstack(dyn_state)
                    dyn_action = np.vstack(dyn_action)
                    dyn_next_state = np.vstack(dyn_next_state)

                    for_loss = self.forward_dynamics[i_dyn].update(dyn_state, dyn_action, dyn_next_state)
                    inv_loss = self.inverse_dynamics[i_dyn].update(dyn_state, dyn_action, dyn_next_state)
                    # print('for_loss {:.5f} inv_loss {:.5f}'.format(for_loss, inv_loss))

            # trajectory reversal augmentation with dynamics-aware filtering
            reverse_state_idx = []
            reverse_state = []
            reverse_action = []
            reverse_reward = []
            reverse_next_state = []
            reverse_absorbing = []
            for i in self.reversible_indices:
                env_name = self.env_names[i]

                reversal_rm_idx = self.reversible_indices.index(i)
                dynamics_idx = self.env_to_dynamics_dict[i]
                reverse_state_i, reverse_action_i, reverse_reward_i, reverse_next_state_i, reverse_absorbing_i, _ = self._reversal_replay_memory[reversal_rm_idx].get(self._batch_size())
                # print('env reverse_reward_i', reverse_reward_i)
                
                # generate reverse action by inverse dynamics
                reverse_step_state_i = np.array([s[0:18] for s in reverse_state_i])
                reverse_step_next_state_i = np.array([s[0:18] for s in reverse_next_state_i])
                reverse_action_i = self.inverse_dynamics[dynamics_idx](reverse_step_state_i, reverse_step_next_state_i).detach().cpu().numpy()
                
                # relabel the reward of (reverse_state, reverse_action, reverse_next_state)
                reverse_reward_i = np.array([state_reward(env_name, reverse_next_state_i[i_transition], reverse_action_i[i_transition]) for i_transition in range(len(reverse_action_i))])
                # print('computed reverse_reward_i', reverse_reward_i)

                # filter out fake transition by forward dynamics
                predicted_reverse_step_next_state_i = self.forward_dynamics[dynamics_idx](reverse_step_state_i, reverse_action_i).detach().cpu().numpy()
                predict_error = np.linalg.norm(reverse_step_next_state_i[:, 4:7] - predicted_reverse_step_next_state_i[:, 4:7], axis=1)
                obj_error = np.linalg.norm(reverse_step_state_i[:, 4:7] - reverse_step_next_state_i[:, 4:7], axis=1) 
                filter_indices = np.logical_or(predict_error < 0.01, obj_error == 0)
                self.reversible_ratio.append(np.mean(filter_indices))
                
                # keep dynamics-consistent transitions
                reverse_state_idx.append(np.ones(np.sum(filter_indices), dtype=np.int64) * i)
                reverse_state.append(reverse_state_i[filter_indices])
                reverse_action.append(reverse_action_i[filter_indices])
                reverse_reward.append(reverse_reward_i[filter_indices])
                reverse_next_state.append(reverse_next_state_i[filter_indices])
                reverse_absorbing.append(reverse_absorbing_i[filter_indices])

            reverse_state_idx = np.hstack(reverse_state_idx)
            reverse_state = np.vstack(reverse_state)
            reverse_action = np.vstack(reverse_action)
            reverse_reward = np.hstack(reverse_reward)
            reverse_next_state = np.vstack(reverse_next_state)
            reverse_absorbing = np.hstack(reverse_absorbing)

            state_idx = np.concatenate([state_idx, reverse_state_idx])
            state = np.vstack([state, reverse_state])
            action = np.vstack([action, reverse_action])
            reward = np.concatenate([reward, reverse_reward])
            next_state = np.vstack([next_state, reverse_next_state])
            absorbing = np.concatenate([absorbing, reverse_absorbing])

            # print('-'*10, 'after aug')
            # print('np.shape(state_idx)',np.shape(state_idx))
            # print('np.shape(state)',np.shape(state))
            # print('np.shape(action)', np.shape(action))
            # print('np.shape(reward)', np.shape(reward))
            # print('np.shape(next_state)', np.shape(next_state))
            # print('np.shape(absorbing)', np.shape(absorbing))

            if self._replay_memory[0].size > self._warmup_transitions(): #first or any
                action_new, log_prob = self.policy.compute_action_and_log_prob_t([state_idx, state])
                loss = self._loss(state, action_new, state_idx, log_prob)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach(), state_idx)

            q_next = self._next_q(next_state, absorbing, state_idx)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q, c = state_idx,
                                          **self._critic_fit_params)
            # TODO: double check this step
            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)
           
    def _loss(self, state, action_new, state_idx, log_prob):

        q_0 = self._critic_approximator(state, action_new, c = state_idx,
                                        output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new, c = state_idx,
                                        output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return (self._alpha(state_idx) * log_prob - q).mean()

    def _update_alpha(self, log_prob, state_idx):
        alpha_loss = - (self._log_alpha_disentangled(state_idx) * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

        return alpha_loss
    
    def _next_q(self, next_state, absorbing, state_idx):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.
        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.
        """

        a, log_prob_next = self.policy.compute_action_and_log_prob([state_idx, next_state])

        q = self._target_critic_approximator.predict(
            next_state, a, c = state_idx, prediction='min') - self._alpha_np(state_idx) * log_prob_next
        q *= 1 - absorbing

        return q
    
    # TODO: improve
    def _log_alpha_disentangled(self, c):
        log_alpha = torch.zeros(size=(c.shape[0],))
        c = c.astype(int)

        if self.policy.use_cuda:
            log_alpha = log_alpha.cuda()

        for _, ci in enumerate(np.unique(c)):
            ci_idx = np.argwhere(c == ci).ravel()
            log_alpha_i = self._log_alpha[ci]
            log_alpha[ci_idx] = log_alpha_i

        return log_alpha
    
    # TODO: improve  
    def _alpha(self, c):
        
        log_alpha = self._log_alpha_disentangled(c)
        return log_alpha.exp()

    def _alpha_np(self, c):
        return self._alpha(c).detach().cpu().numpy()

    def get_log_alpha(self, c):
        '''
            get the value of log_alpha of context c for logging
        '''
        return self._log_alpha[c].detach().cpu().numpy()

    def set_log_alpha(self, log_alpha):
        for c in range(self._n_contexts()):
            self._log_alpha[c] = log_alpha[c]