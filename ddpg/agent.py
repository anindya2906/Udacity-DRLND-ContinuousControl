import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .models import Actor, Critic
from .buffers import ReplayBuffer


class Agent():
    """Interacts with and learn from environment"""

    def __init__(self, state_size, action_size, config):
        """initialize an agent object
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            config (object): configurations
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.seed = random.seed(self.config['seed'])
        self.time_step = 0
        
        # Action networks; Local and Target
        self.actor_local = Actor(self.state_size, self.action_size, self.config['seed']).to(self.config['device'])
        self.actor_target = Actor(self.state_size, self.action_size, self.config['seed']).to(self.config['device'])
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config['lr_actor'])

        # Critic networks; Local and Target
        self.critic_local = Critic(self.state_size, self.action_size, self.config['seed']).to(self.config['device'])
        self.critic_target = Critic(self.state_size, self.action_size, self.config['seed']).to(self.config['device'])
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config['lr_critic'])

        # Noise Process
        self.noise = OUNiose(self.action_size, self.config["seed"])

        # Replay memory
        self.memory = ReplayBuffer(self.config['buffer_size'], self.config['batch_size'], self.config['seed'], self.config['device'])
    
    def step(self, state, action, reward, next_state, done):
        """Store experience and learn from a random batch"""
        self.memory.add(state, action, reward, next_state, done)

        self.time_step = (self.time_step + 1) % self.config['update_every']
        if self.time_step == 0:
            if len(self.memory) > self.config['batch_size']:
                experiences = self.memory.sample()
                self.learn(experiences, self.config['gamma'])

    def act(self, state, add_noise=True):
        """Returns an action given a state based on the current policy"""
        state = torch.from_numpy(state).float().to(self.config['device'])
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples
        Q_targets = reward + gamma * critic_target(next_state, actor_target(next_state))
        where
            actor_target(sate) -> action
            critic_target(state, action) -> Q-value
        
        Params
        ======
            experiences: (states, actions, rewrds, next_states, dones)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update Critic
        #-----------------------
        # Get predicted next state actions an Q-values from target models
        actions_next = self.actor_target(next_states)
        Q_target_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_target_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        #-------------------------
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_upate(self.critic_local, self.critic_target, self.config['tau'])
        self.soft_upate(self.actor_local, self.actor_target, self.config['tau'])


    def soft_upate(self, local_network, target_network, tau):
        """Soft update model parameters
        Params
        ======
            local_network: Pytorch model (weights will be copied from)
            target_network: Pytorch model (weights will be copied to)
            tau (float): Interpolation parameter
        """
        for target_params, local_params in zip(target_network.parameters(), local_network.parameters()):
            target_params.data.copy_(tau*local_params.data + (1.0-tau)*target_params.data)

class OUNiose():
    """Ornstein-Uhlenbeck process"""
    
    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        """Initiallize parameters and noise process"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (noise) to mean (mu)"""
        self.state = copy.copy(self.mu)
    
    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state