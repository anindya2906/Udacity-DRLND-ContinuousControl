from ast import parse
import gym
import argparse
import torch
from ddpg.agent import Agent


class DDPGApp():
    """Application class to solve various environments 
    using Deep Deterministic Poicy Gradient (DDPG) algorithm"""

    def __init__(self, env_name, configs):
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size, configs)

    def run_training(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Deterministic Policy Gradient (DDPG)")
    parser.add_argument("--env_name", type=str, help="Name of the environment")
    # parser.add_argument()
    args = parser.parse_args()
    configs = {
        'seed': 0,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'buffer_size': 100000,
        'batch_size': 128,
        'gamma': 0.99,
        'tau': 1e-3
    }

    ddpg_app = DDPGApp(args.env_name, configs)

