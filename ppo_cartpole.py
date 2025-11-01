import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__() #init nn.Module
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
    
    
    def act(self, state):
        shared_features = self.shared(state)
        action_probs = torch.softmax(self.actor(shared_features), dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action, action_log_prob
    
    def evaluate(self, state, action):
        shared_features = self.shared(state)
        action_probs = torch.softmax(self.actor(shared_features), dim=-1) #maak het een prob function
        dist = Categorical(action_probs) #maak prob function van de action probs

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(shared_features)
        
        return action_log_probs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 K_epochs=4, value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_log_prob = self.policy_old.act(state)
        return action.item(), action_log_prob.item()
    
    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory['rewards']), reversed(memory['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.FloatTensor(np.array(memory['states']))
        old_actions = torch.LongTensor(np.array(memory['actions']))
        old_logprobs = torch.FloatTensor(np.array(memory['logprobs']))
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + self.value_coef * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        

def train_ppo():
    # Environment setup
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Hyperparameters
    max_episodes = 2000
    max_timesteps = 500
    update_timestep = 2000
    
    # PPO agent
    ppo = PPO(state_dim, action_dim)
    
    # Training loop
    time_step = 0
    episode_rewards = []
    
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
        
        for t in range(max_timesteps):
            time_step += 1
            
            # Select action
            action, action_log_prob = ppo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store data
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(action_log_prob)
            memory['rewards'].append(reward)
            memory['is_terminals'].append(done)
            
            state = next_state
            episode_reward += reward
            
            # Update PPO agent
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Avg Reward (last 10): {avg_reward:.2f}")
            
            # Check if solved
            if avg_reward >= 475:
                print(f"Solved in {episode} episodes!")
                break
    
    env.close()
    return episode_rewards


if __name__ == "__main__":
    rewards = train_ppo()
    print(f"Training completed. Final average reward: {np.mean(rewards[-100:]):.2f}")