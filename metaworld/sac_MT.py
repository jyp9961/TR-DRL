import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os

import utils
import time

class ReplayBuffer(object):
    def __init__(self, max_size, state_size, agent_state_size, env_state_size, action_size):
        # define memory size and counter of the replay buffer
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *[state_size]))
        self.agent_state_memory = np.zeros((self.mem_size, *[agent_state_size]))
        self.env_state_memory = np.zeros((self.mem_size, *[env_state_size]))
        self.new_state_memory = np.zeros((self.mem_size, *[state_size]))
        self.new_agent_state_memory = np.zeros((self.mem_size, *[agent_state_size]))
        self.new_env_state_memory = np.zeros((self.mem_size, *[env_state_size]))
        self.action_memory = np.zeros((self.mem_size, *[action_size]))
        self.reward_memory = np.zeros(self.mem_size)
        self.not_dones_memory = np.zeros(self.mem_size)
    
    def store_transition(self, state, agent_state, env_state, action, reward, state_, agent_state_, env_state_, done):
        # store one transition into the replay buffer
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.agent_state_memory[index] = agent_state
        self.env_state_memory[index] = env_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.new_agent_state_memory[index] = agent_state_
        self.new_env_state_memory[index] = env_state_
        self.not_dones_memory[index] = 1-int(done)
        self.mem_cntr += 1

    def sample(self, batch_size):
        # randomly sample and return batch_size number of transitions
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        agent_states = self.agent_state_memory[batch]
        env_states = self.env_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        new_agent_states = self.new_agent_state_memory[batch]
        new_env_states = self.new_env_state_memory[batch]
        not_dones = self.not_dones_memory[batch]

        return states, agent_states, env_states, actions, rewards, new_states, new_agent_states, new_env_states, not_dones
    
    def save(self, replay_buffer_dir):
        pass

class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, state_size, env_embedding_size, action_size, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()
        self.state_size = state_size
        self.env_embedding_size = env_embedding_size
        self.action_size = action_size
        self.log_std_bounds = log_std_bounds
        # input: [state, env_embedding]
        # output: [mu, std]
        self.input_size = state_size + env_embedding_size
        self.output_size = self.action_size * 2
        self.trunk = utils.mlp(self.input_size, hidden_dim, self.output_size, hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, state, env_embedding):
        assert state.size(0) == env_embedding.size(0)
        len_output = state.size(0)
        state = torch.cat([state, env_embedding], dim=-1) #w/ env_embedding
        mu, log_std = self.trunk(state).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist

class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, state_size, env_embedding_size, action_size, hidden_dim, hidden_depth):
        super().__init__()
        self.state_size = state_size
        self.env_embedding_size = env_embedding_size
        self.action_size = action_size
        # input: [state, env_embedding, action]
        # output (for each Q): [Q]
        self.input_size = state_size + env_embedding_size + action_size
        self.output_size = 1
        self.Q1 = utils.mlp(self.input_size, hidden_dim, self.output_size, hidden_depth)
        self.Q2 = utils.mlp(self.input_size, hidden_dim, self.output_size, hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs, env_embedding, action):
        assert obs.size(0) == env_embedding.size(0) == action.size(0)
        len_output = obs.size(0)
        obs_action = torch.cat([obs, env_embedding, action], dim=-1) #w/ env_embedding
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
    
        return q1, q2

class Env_Potential(nn.Module):
    """Env Potential Function Estimator"""
    def __init__(self, env_state_size, env_embedding_size, hidden_dim, hidden_depth):
        super().__init__()
        self.env_state_size = env_state_size
        self.env_embedding_size = env_embedding_size
        # input: [env_state, env_embedding]
        # output: potential
        self.input_size = env_state_size + env_embedding_size
        self.output_size = 1
        self.potential_estimator = utils.mlp(self.input_size, hidden_dim, self.output_size, hidden_depth, nn.Sigmoid())

        self.apply(utils.weight_init)

    def forward(self, env_state, env_embedding):
        assert env_state.size(0) == env_embedding.size(0)
        len_output = env_state.size(0)
        env_state = torch.cat([env_state, env_embedding], dim=-1)
        potential = self.potential_estimator(env_state)
            
        return potential

class ForwardDynamics(nn.Module):
    """Forward Dynamics network"""
    def __init__(self, state_size, env_embedding_size, action_size, hidden_dim, hidden_depth, device):
        super().__init__()
        self.input_dims = state_size + env_embedding_size + action_size
        self.output_dims = state_size

        self.for_dynamics = utils.mlp(self.input_dims, hidden_dim, self.output_dims, hidden_depth, output_mod = nn.Tanh())
        self.apply(utils.weight_init)
        self.device = device
        
    def forward(self, state, env_embedding, action):
        if isinstance(state, np.ndarray): state = torch.FloatTensor(state).to(self.device)
        if isinstance(env_embedding, np.ndarray): env_embedding = torch.FloatTensor(env_embedding).to(self.device)
        if isinstance(action, np.ndarray): action = torch.FloatTensor(action).to(self.device)
        assert state.size(0) == env_embedding.size(0) == action.size(0)

        state_env_actions = torch.cat([state, env_embedding, action], dim=-1)
        # the forward dynamics model predicts the difference between states and new_states
        diff_state = self.for_dynamics(state_env_actions)
        new_state = state + diff_state

        return new_state

class InverseDynamics(nn.Module):
    """Inverse Dynamics network"""
    def __init__(self, state_size, env_embedding_size, action_size, hidden_dim, hidden_depth, device):
        super().__init__()
        self.input_dims = state_size * 2 + env_embedding_size
        self.output_dims = action_size

        self.inv_dynamics = utils.mlp(self.input_dims, hidden_dim, self.output_dims, hidden_depth, output_mod = nn.Tanh())
        self.apply(utils.weight_init)
        self.device = device
        
    def forward(self, state, env_embedding, next_state):
        if isinstance(state, np.ndarray): state = torch.FloatTensor(state).to(self.device)
        if isinstance(env_embedding, np.ndarray): env_embedding = torch.FloatTensor(env_embedding).to(self.device)
        if isinstance(next_state, np.ndarray): next_state = torch.FloatTensor(next_state).to(self.device)
        assert state.size(0) == env_embedding.size(0) == next_state.size(0)

        states = torch.cat([state, env_embedding, next_state-state], dim=-1)
        actions = self.inv_dynamics(states)

        return actions

class SACAgent(object):
    """Soft Actor-Critic for n tasks"""
    def __init__(self, state_size, agent_state_size, env_state_size, action_size, action_range, device, discount, init_temperature, actor_lr, critic_lr, potential_lr, for_lr, inv_lr, alpha_lr, actor_update_frequency, critic_tau, critic_target_update_frequency, batch_size, hidden_dim=512, hidden_depth=2, log_std_bounds=[-10, 2], demo_replay_buffer_size=100000, replay_buffer_size=100000, num_envs=10, env_names=[], env_embedding_size=50, env_embeddings=[], env_dynamics_indices=[]):
        self.state_size, self.agent_state_size, self.env_state_size, self.action_size = state_size, agent_state_size, env_state_size, action_size
        self.action_range, self.device, self.discount, self.init_temperature, self.actor_lr, self.critic_lr = action_range, device, discount, init_temperature, actor_lr, critic_lr
        self.actor_update_frequency, self.critic_tau, self.critic_target_update_frequency = actor_update_frequency, critic_tau, critic_target_update_frequency
        self.batch_size, self.hidden_dim, self.hidden_depth = batch_size, hidden_dim, hidden_depth
        self.log_std_bounds = log_std_bounds
        self.demo_replay_buffer_size, self.replay_buffer_size = demo_replay_buffer_size, replay_buffer_size
        self.num_envs = num_envs
        self.env_names = env_names
        self.env_embedding_size = env_embedding_size
        self.env_embeddings = env_embeddings
        self.env_dynamics_indices = env_dynamics_indices
        
        # store agent's interactions with envs
        self.replay_buffers = {}
        
        # store demos from experts
        self.demo_replay_buffers = {}
        
        # store successful trajectories
        self.success_replay_buffers = {}
        
        # store successful trajectories from the other agent
        self.reversal_replay_buffers = {}
        
        # store env states of successful trajectories from the other agent
        self.reverse_replay_buffers = {}
        
        for i_env in range(num_envs):
            self.replay_buffers[i_env] = ReplayBuffer(self.replay_buffer_size, self.state_size, self.agent_state_size, self.env_state_size, self.action_size)
            self.demo_replay_buffers[i_env] = ReplayBuffer(self.demo_replay_buffer_size, self.state_size, self.agent_state_size, self.env_state_size, self.action_size)
            self.success_replay_buffers[i_env] = ReplayBuffer(self.replay_buffer_size, self.state_size, self.agent_state_size, self.env_state_size, self.action_size)
            self.reversal_replay_buffers[i_env] = ReplayBuffer(self.replay_buffer_size, self.state_size, self.agent_state_size, self.env_state_size, self.action_size)
            self.reverse_replay_buffers[i_env] = ReplayBuffer(self.replay_buffer_size, self.state_size, self.agent_state_size, self.env_state_size, self.action_size)
        
        # actor, critic, potential model
        self.actor = Actor(self.state_size, self.env_embedding_size, self.action_size, self.hidden_dim, self.hidden_depth, self.log_std_bounds).to(self.device)
        print('Actor', self.actor)
        self.critic = Critic(self.state_size, self.env_embedding_size, self.action_size, self.hidden_dim, self.hidden_depth).to(self.device)
        print('Critic', self.critic)
        self.critic_target = Critic(self.state_size, self.env_embedding_size, self.action_size, self.hidden_dim, self.hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.forward_potential = Env_Potential(self.env_state_size, self.env_embedding_size, self.hidden_dim, self.hidden_depth).to(self.device)
        print('Forward potential', self.forward_potential)
        self.reverse_potential = Env_Potential(self.env_state_size, self.env_embedding_size, self.hidden_dim, self.hidden_depth).to(self.device)
        print('Reverse potential', self.reverse_potential)

        # forward and inverse dynamics, log_alpha
        self.forward_dynamics = ForwardDynamics(self.state_size, self.env_embedding_size, self.action_size, self.hidden_dim, self.hidden_depth, self.device).to(self.device)
        self.inverse_dynamics = InverseDynamics(self.state_size, self.env_embedding_size, self.action_size, self.hidden_dim, self.hidden_depth, self.device).to(self.device)
        
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_size

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.forward_potential_optimizer = torch.optim.Adam(self.forward_potential.parameters(), lr=potential_lr)
        self.reverse_potential_optimizer = torch.optim.Adam(self.reverse_potential.parameters(), lr=potential_lr)
        self.forward_dynamics_optimizer = torch.optim.Adam(self.forward_dynamics.parameters(), lr=for_lr)
        self.inverse_dynamics_optimizer = torch.optim.Adam(self.inverse_dynamics.parameters(), lr=inv_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
    
    def eval(self):
        self.actor.eval()
        self.critic.eval()
    
    def sample_replay_buffer(self, replay_buffer, batch_size):
        states, agent_states, env_states, actions, rewards, new_states, new_agent_states, new_env_states, not_dones = replay_buffer.sample(batch_size)
        states = torch.as_tensor(states, device=self.device).float()
        agent_states = torch.as_tensor(agent_states, device=self.device).float()
        env_states = torch.as_tensor(env_states, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device).float()
        rewards = torch.as_tensor(rewards, device=self.device).float()
        new_states = torch.as_tensor(new_states, device=self.device).float()
        new_agent_states = torch.as_tensor(new_agent_states, device=self.device).float()
        new_env_states = torch.as_tensor(new_env_states, device=self.device).float()
        not_dones = torch.as_tensor(not_dones, device=self.device).float()

        return states, agent_states, env_states, actions, rewards, new_states, new_agent_states, new_env_states, not_dones

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, env_embedding, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        env_embedding = torch.FloatTensor(env_embedding).to(self.device)
        env_embedding = env_embedding.unsqueeze(0)
        dist = self.actor(obs, env_embedding)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return utils.to_np(action[0])

    def target_Qs(self, reward, next_obs, env_embedding, not_done):
        with torch.no_grad():
            dist = self.actor(next_obs, env_embedding)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            
            target_Q1, target_Q2 = self.critic_target(next_obs, env_embedding, next_action)
            
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            
            reward = reward.reshape(-1,1)
            not_done = not_done.reshape(-1,1)
            target_Q = reward + self.discount * not_done * target_V
            
        return target_Q

    def critic_loss(self, target_Q, current_Q1, current_Q2):
        Q1_loss = F.mse_loss(current_Q1, target_Q)
        Q2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss =  Q1_loss + Q2_loss
        
        return critic_loss

    def actor_losses(self, obs, env_embedding, action):
        dist = self.actor(obs, env_embedding)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        actor_Q1, actor_Q2 = self.critic(obs, env_embedding, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        return actor_loss, log_prob

    def update_critic(self, obs, env_embedding, action, reward, next_obs, not_done):
        target_Q = self.target_Qs(reward, next_obs, env_embedding, not_done)
        
        current_Q1, current_Q2 = self.critic(obs, env_embedding, action)

        critic_loss = self.critic_loss(target_Q, current_Q1, current_Q2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return utils.to_np(critic_loss)

    def update_actor_and_alpha(self, obs, env_embedding, action):
        actor_loss, log_prob = self.actor_losses(obs, env_embedding, action)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return utils.to_np(actor_loss), utils.to_np(alpha_loss), log_prob

    def update_potential(self, type, update=True):
        all_reward_loss = 0
        num_update_envs = 0
        for i_env in range(self.num_envs):
            success_replay_buffer = self.success_replay_buffers[i_env]
            reverse_replay_buffer = self.reverse_replay_buffers[i_env]
            env_embedding_tensor = torch.FloatTensor(self.env_embeddings[i_env]).to(self.device)

            if type=='forward':
                batch_size = min(success_replay_buffer.mem_cntr, int(self.batch_size // 2))
                _, _, _, _, reward, _, _, new_env_state, _ = self.sample_replay_buffer(success_replay_buffer, batch_size)
                env_embedding_tensor = env_embedding_tensor.repeat(len(new_env_state), 1)
                predicted_reward = self.forward_potential(new_env_state, env_embedding_tensor)

            if type=='reverse':
                batch_size = min(reverse_replay_buffer.mem_cntr, int(self.batch_size // 2))
                _, _, _, _, reward, _, _, new_env_state, _ = self.sample_replay_buffer(reverse_replay_buffer, batch_size)
                env_embedding_tensor = env_embedding_tensor.repeat(len(new_env_state), 1)
                predicted_reward = self.reverse_potential(new_env_state, env_embedding_tensor)

            reward = reward.reshape(-1, 1)
            if len(reward) > 0:
                reward_loss = F.mse_loss(predicted_reward, reward)
                num_update_envs += 1
                all_reward_loss += reward_loss
        
        all_reward_loss = all_reward_loss / num_update_envs

        if update:
            if type=='forward':
                self.forward_potential_optimizer.zero_grad()
                all_reward_loss.backward()
                self.forward_potential_optimizer.step()
            if type=='reverse':
                self.reverse_potential_optimizer.zero_grad()
                all_reward_loss.backward()
                self.reverse_potential_optimizer.step()
    
        return utils.to_np(all_reward_loss)

    def update_forward_dynamics(self):
        all_for_loss = 0
        num_update_envs = 0
        for i_env in range(self.num_envs):
            env_embedding_tensor = torch.FloatTensor(self.env_embeddings[self.env_dynamics_indices[i_env]]).to(self.device)
            obs, action, next_obs = [], [], []
        
            demo_batch_size = min(self.demo_replay_buffers[i_env].mem_cntr, int(self.batch_size // 2))
            normal_batch_size = int(self.batch_size)

            normal_obs, normal_agent_obs, normal_env_obs, normal_action, normal_reward, normal_next_obs, normal_next_agent_obs, normal_next_env_obs, normal_not_done = self.sample_replay_buffer(self.replay_buffers[i_env], normal_batch_size)
            demo_obs, demo_agent_obs, demo_env_obs, demo_action, demo_reward, demo_next_obs, demo_next_agent_obs, demo_next_env_obs, demo_not_done = self.sample_replay_buffer(self.demo_replay_buffers[i_env], demo_batch_size)
        
            obs = torch.cat([normal_obs, demo_obs], dim=0)
            action = torch.cat([normal_action, demo_action], dim=0)
            next_obs = torch.cat([normal_next_obs, demo_next_obs], dim=0)
            env_embedding_tensor = env_embedding_tensor.repeat(len(obs), 1)

            predicted_next_obs = self.forward_dynamics(obs, env_embedding_tensor, action)
            if len(next_obs) > 0:
                for_loss = F.mse_loss(next_obs, predicted_next_obs)
                num_update_envs += 1
                all_for_loss += for_loss
                
        all_for_loss = all_for_loss / num_update_envs

        self.forward_dynamics_optimizer.zero_grad()
        all_for_loss.backward()
        self.forward_dynamics_optimizer.step()

        return utils.to_np(all_for_loss)

    def update_inverse_dynamics(self):
        all_inv_loss = 0
        num_update_envs = 0
        for i_env in range(self.num_envs):
            env_embedding_tensor = torch.FloatTensor(self.env_embeddings[self.env_dynamics_indices[i_env]]).to(self.device)
            obs, action, next_obs = [], [], []
        
            demo_batch_size = min(self.demo_replay_buffers[i_env].mem_cntr, int(self.batch_size // 2))
            normal_batch_size = int(self.batch_size)

            normal_obs, normal_agent_obs, normal_env_obs, normal_action, normal_reward, normal_next_obs, normal_next_agent_obs, normal_next_env_obs, normal_not_done = self.sample_replay_buffer(self.replay_buffers[i_env], normal_batch_size)
            demo_obs, demo_agent_obs, demo_env_obs, demo_action, demo_reward, demo_next_obs, demo_next_agent_obs, demo_next_env_obs, demo_not_done = self.sample_replay_buffer(self.demo_replay_buffers[i_env], demo_batch_size)
        
            obs = torch.cat([normal_obs, demo_obs], dim=0)
            action = torch.cat([normal_action, demo_action], dim=0)
            next_obs = torch.cat([normal_next_obs, demo_next_obs], dim=0)
            env_embedding_tensor = env_embedding_tensor.repeat(len(obs), 1)

            predicted_action = self.inverse_dynamics(obs, env_embedding_tensor, next_obs)
            if len(next_obs) > 0:
                inv_loss = F.mse_loss(action, predicted_action)
                num_update_envs += 1
                all_inv_loss += inv_loss
                
        all_inv_loss = all_inv_loss / num_update_envs

        self.inverse_dynamics_optimizer.zero_grad()
        all_inv_loss.backward()
        self.inverse_dynamics_optimizer.step()

        return utils.to_np(all_inv_loss)

    def update(self, step, use_forward_reward, use_reversed_reward, reward_model_type, reward_model_max_value, horizon, use_reversed_transition, all_thresholds, all_state_max, all_state_min, filter_transition, filter_type):
        for i_env in range(self.num_envs):
            env_name = self.env_names[i_env]
            thresholds = all_thresholds[i_env]
            state_max = all_state_max[i_env]
            state_min = all_state_min[i_env]

            replay_buffer = self.replay_buffers[i_env]
            demo_replay_buffer = self.demo_replay_buffers[i_env]
            reversal_replay_buffer = self.reversal_replay_buffers[i_env]

            normal_batch_size = self.batch_size
            demo_batch_size = min(demo_replay_buffer.mem_cntr, int(self.batch_size // 2))
            reverse_batch_size = min(reversal_replay_buffer.mem_cntr, int(self.batch_size // 2))

            normal_obs, normal_agent_obs, normal_env_obs, normal_action, normal_reward, normal_next_obs, normal_next_agent_obs, normal_next_env_obs, normal_not_done = self.sample_replay_buffer(replay_buffer, normal_batch_size)
            demo_obs, demo_agent_obs, demo_env_obs, demo_action, demo_reward, demo_next_obs, demo_next_agent_obs, demo_next_env_obs, demo_not_done = self.sample_replay_buffer(demo_replay_buffer, demo_batch_size)
            reverse_obs, reverse_agent_obs, reverse_env_obs, reverse_action, reverse_reward, reverse_next_obs, reverse_next_agent_obs, reverse_next_env_obs, reverse_not_done = self.sample_replay_buffer(reversal_replay_buffer, reverse_batch_size)

            obs = torch.cat([normal_obs, demo_obs], dim=0)
            env_obs = torch.cat([normal_env_obs, demo_env_obs], dim=0)
            action = torch.cat([normal_action, demo_action], dim=0)
            reward = torch.cat([normal_reward, demo_reward], dim=0)
            next_obs = torch.cat([normal_next_obs, demo_next_obs], dim=0)
            next_env_obs = torch.cat([normal_next_env_obs, demo_next_env_obs], dim=0)
            not_done = torch.cat([normal_not_done, demo_not_done], dim=0)

            reversible_ratio = 0
            if use_reversed_transition:
                dynamics_env_embedding_tensor = torch.FloatTensor(self.env_embeddings[self.env_dynamics_indices[i_env]]).to(self.device).repeat(len(reverse_obs), 1)
                with torch.no_grad():
                    predicted_reverse_action = self.inverse_dynamics(reverse_obs, dynamics_env_embedding_tensor, reverse_next_obs)
                    predicted_reverse_next_obs = self.forward_dynamics(reverse_obs, dynamics_env_embedding_tensor, predicted_reverse_action)
                    reversible_indices = filter_transition(reverse_obs, reverse_next_obs, predicted_reverse_next_obs, env_name, thresholds, state_max, state_min, filter_type)
                    reversible_ratio = utils.to_np(torch.mean(reversible_indices.float()))

                reversed_obs = reverse_obs[reversible_indices]
                reversed_env_obs = reverse_env_obs[reversible_indices]
                reversed_next_obs = reverse_next_obs[reversible_indices]
                reversed_next_env_obs = reverse_next_env_obs[reversible_indices]
                reversed_action = predicted_reverse_action[reversible_indices]
                reversed_reward = reverse_reward[reversible_indices]
                reversed_not_done = reverse_not_done[reversible_indices]

                obs = torch.cat([obs, reversed_obs], dim=0)
                env_obs = torch.cat([env_obs, reversed_env_obs], dim=0)
                action = torch.cat([action, reversed_action], dim=0)
                reward = torch.cat([reward, reversed_reward], dim=0)
                next_obs = torch.cat([next_obs, reversed_next_obs], dim=0)
                next_env_obs = torch.cat([next_env_obs, reversed_next_env_obs], dim=0)
                not_done = torch.cat([not_done, reversed_not_done], dim=0)

            if reward_model_type == 'potential':
                if use_forward_reward or use_reversed_reward:
                    # relabel reward
                    env_embedding_tensor = torch.FloatTensor(self.env_embeddings[i_env]).to(self.device).repeat(len(obs), 1)
                    
                    forward_env_potential = self.forward_potential(env_obs, env_embedding_tensor).reshape(-1)
                    forward_env_next_potential = self.forward_potential(next_env_obs, env_embedding_tensor).reshape(-1)
                    
                    reverse_env_potential = self.reverse_potential(env_obs, env_embedding_tensor).reshape(-1)
                    reverse_env_next_potential = self.reverse_potential(next_env_obs, env_embedding_tensor).reshape(-1)
                    
                    forward_env_reward = torch.clip((forward_env_next_potential - forward_env_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value)
                    reverse_env_reward = torch.clip((reverse_env_next_potential - reverse_env_potential) * reward_model_max_value * horizon, min=-reward_model_max_value, max=reward_model_max_value)

                    env_reward = 0
                    if use_forward_reward: env_reward += forward_env_reward
                    if use_reversed_reward: env_reward += reverse_env_reward
                    if use_forward_reward and use_reversed_reward: env_reward /= 2
                    reward = env_reward * (reward == 0) + reward
            
            env_embedding_tensor = torch.FloatTensor(self.env_embeddings[i_env]).to(self.device).repeat(len(obs), 1)
            
            if i_env == 0:
                all_obs = obs
                all_env_embedding = env_embedding_tensor
                all_action = action
                all_reward = reward
                all_next_obs = next_obs
                all_not_done = not_done
            else:
                all_obs = torch.cat([all_obs, obs], dim=0)
                all_env_embedding = torch.cat([all_env_embedding, env_embedding_tensor], dim=0)
                all_action = torch.cat([all_action, action], dim=0)
                all_reward = torch.cat([all_reward, reward], dim=0)
                all_next_obs = torch.cat([all_next_obs, next_obs], dim=0)
                all_not_done = torch.cat([all_not_done, not_done], dim=0)

        critic_loss = self.update_critic(all_obs, all_env_embedding, all_action, all_reward, all_next_obs, all_not_done)

        actor_loss = 0
        alpha_loss = 0
        log_prob = 0
        # update actor
        if step % self.actor_update_frequency == 0:
            actor_loss, alpha_loss, log_prob = self.update_actor_and_alpha(all_obs, all_env_embedding, all_action)
            
        # update target networks
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        if critic_loss > 1000:
            self.critic.apply(utils.weight_init)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor.apply(utils.weight_init)

        return critic_loss, actor_loss, alpha_loss, float(utils.to_np(self.alpha)), log_prob

    def save(self):
        pass
    
    def load(self):
        pass