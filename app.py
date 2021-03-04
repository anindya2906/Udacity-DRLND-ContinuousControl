import gym
import argparse
import torch
import numpy as np
from ddpg.agent import Agent
from collections import deque


class DDPGApp():
    """Application class to solve various environments 
    using Deep Deterministic Poicy Gradient (DDPG) algorithm"""

    def __init__(self, env_name, configs):
        self.env = gym.make(env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.agent = Agent(self.state_size, self.action_size, configs)

    def run_training(self, n_episodes=1000, max_t=300, print_every=100, solving_score=30):
        scores_deque = deque(maxlen=print_every)
        scores = []
        for i_episode in range(1, n_episodes+1):
            state = self.env.reset()
            score = 0
            for t in range(max_t):
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            print(f"\rEpisode: {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")
            # if i_episode % print_every == 0:
            #     print(f"\rEpisode: {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}")
            if np.mean(scores_deque) >= solving_score:
                torch.save(self.agent.actor_local.state_dict(), 'checkpoint_actor.pt')
                torch.save(self.agent.critic_local.state_dict(), 'checkpoint_critic.pt')
                print(f"Episode solved in {i_episode-100} episodes with average score {np.mean(scores_deque)}")
                break
        return score


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
    scores = ddpg_app.run_training()

