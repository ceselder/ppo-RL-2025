from Environment import BaseEnvironment
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
import numpy as np
from nn_utils import GenericNeuralNetwork
import random
import pandas as pd
import itertools

import matplotlib.pyplot as plt

import pickle

with open('public_data_dict.pkl', 'rb') as f:
    data = pickle.load(f)

class PPOHyperparameters():

    def __init__(self, num_batches: int, sampling_variance: float, gamma: float, updates_per_iteration: int, clip_value: float, actor_lr_start: float, actor_lr_end: float, critic_lr_start: float, critic_lr_end: float, cutoff: int, use_entropy: bool = False, beta_entropy: float = 0.01):
        self.num_batches = num_batches

        # Variance of the multivariate Normal distribution used to sample exploratory actions
        self.sampling_variance = sampling_variance

        # Discount factor
        self.gamma = gamma

        # Number of updates in each training iteration
        self.updates_per_iteration = updates_per_iteration

        # Epsilon used in the clip for training
        self.clip_value = clip_value

        # Networks Learning Rate
        self.actor_lr_start = actor_lr_start
        self.actor_lr_end = actor_lr_end
        self.critic_lr_start = critic_lr_start
        self.critic_lr_end = critic_lr_end
        self.cutoff = cutoff

        self.use_entropy = use_entropy
        self.beta_entropy = beta_entropy

class PPOAgent:

    def __init__(self, env, state_dim, action_dim, actor: nn.Module, critic: nn.Module, hyperparameters: PPOHyperparameters):

        self.env = env

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hyperparameters = hyperparameters

        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hyperparameters.actor_lr_start)

        self.critic = critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hyperparameters.critic_lr_start)

        self.cov_var = torch.full(size=(self.action_dim,), fill_value = self.hyperparameters.sampling_variance)
        self.cov_mat = torch.diag(self.cov_var) # This will be used to sample actions from the agent distribution. The higher the variance, the wider the gaussian distribution.


    def train(self, num_iterations:int = 200):
        for iteration in range(num_iterations):
            self.update_learning_rate(iteration)
            batch_states, batch_actions, batch_log_probs, batch_cumulative_rewards, batch_rewards = self.rollout()

            V, _, _ = self.calculate_value_logprobs(batch_states=batch_states, batch_actions=batch_actions)
            A_k = batch_cumulative_rewards - V.detach().squeeze(1)
            A_k = torch.clamp(A_k,-1,1)
            

            for _ in range(self.hyperparameters.updates_per_iteration):
                current_V, current_log_probs, entropy = self.calculate_value_logprobs(batch_states=batch_states, batch_actions=batch_actions)
                current_V = current_V.squeeze(1)
                probability_ratios = torch.exp(current_log_probs - batch_log_probs)
                if not self.hyperparameters.use_entropy:
                    entropy = torch.tensor(0.0)
                actor_loss = -torch.min(probability_ratios * A_k, torch.clamp(probability_ratios, 1 - self.hyperparameters.clip_value, 1 + self.hyperparameters.clip_value) * A_k).mean() + entropy.mean() * self.hyperparameters.beta_entropy
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                critic_loss = torch.nn.MSELoss()(current_V, batch_cumulative_rewards)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


    def calculate_value_logprobs(self, batch_states, batch_actions):
        V = self.critic(batch_states)
        action_means = self.actor(batch_states)
        dist = MultivariateNormal(action_means, self.cov_mat)
        entropy = dist.entropy()
        log_probs = dist.log_prob(batch_actions)
        V = torch.clamp(V, -1, 1)
        return V, log_probs, entropy

    def sample_action(self, state, explore=False):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(dim=0)
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, self.cov_mat)
        if explore:
            action = dist.sample()
        else:
            action = action_mean
        log_prob = dist.log_prob(action)
        return action, log_prob

    def compute_cumulative_rewards(self, batch_rewards):
        batch_cumulative_rewards = []

        for rewards in batch_rewards:
            discounted_reward = 0
            returns = []
            for r in reversed(rewards):
                discounted_reward = r + self.hyperparameters.gamma * discounted_reward
                returns.insert(0, discounted_reward)
            batch_cumulative_rewards.extend(returns)

        batch_cumulative_rewards = torch.tensor(batch_cumulative_rewards, dtype=torch.float)
        return torch.clamp(batch_cumulative_rewards,-1,1)

    def rollout(self):
        # NOTE: Put the correct sizes as comment here
        batch_states = []       # (num_batches * timesteps_per_episode, state_dim)
        batch_actions = []      # (num_batches * timesteps_per_episode, action_dim)
        batch_log_probs = []    # (num_batches * timesteps_per_episode)
        batch_rewards = []      # (num_batches, timesteps_per_episode)

        for t in range(self.hyperparameters.num_batches):
            self.env.reset(np.random.choice(train_set_indexes))   # Train on a random day from the training set
            rewards_episode = []

            state = self.env.state
            done = self.env.done
            # Exercise: Perform an episode (execution until the done flag variable in the environment becomes True)
            # and fill the lists 'batch_state', 'batch_actions', 'batch_log_probs', and 'batch_rewards' with their
            # respective values obtained from the interaction with the environment.
            while not done:
                batch_states.append(state)
                action, log_prob = self.sample_action(state, explore=True)
                batch_actions.append(action)
                batch_log_probs.append(log_prob.item())
                state, reward, done, _ = self.env.step(self.env.descale_action(action.detach().item()))
                rewards_episode.append(reward)
            batch_rewards.append(rewards_episode)

        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float).unsqueeze(dim=-1)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float)

        batch_cumulative_rewards = self.compute_cumulative_rewards(batch_rewards=batch_rewards)     # (timesteps_per_episode)

        return batch_states, batch_actions, batch_log_probs, batch_cumulative_rewards, batch_rewards
    
    def update_learning_rate(self, epoch):
        epoch = min(epoch, self.hyperparameters.cutoff)
        new_actor_lr = self.hyperparameters.actor_lr_start-(epoch/self.hyperparameters.cutoff)*(self.hyperparameters.actor_lr_start-self.hyperparameters.actor_lr_end)
        new_critic_lr = self.hyperparameters.critic_lr_start-(epoch/self.hyperparameters.cutoff)*(self.hyperparameters.critic_lr_start-self.hyperparameters.critic_lr_end)
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_actor_lr

        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_critic_lr

    def save_agent(self):
        with open('actor.pickle', 'wb') as handle:
            pickle.dump(self.actor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_agent(self):
        with open('actor.pickle', 'rb') as handle:
            self.actor = pickle.load(handle)
    


train_length = int(len(data) * 0.8)

train_set_indexes = list(range(0, train_length))
test_set_indexes = list(range(train_length, len(data)))

environment = BaseEnvironment(data = data, power_capacity=4, energy_capacity=8)

state_dim = 5       # Dimension of each state, do not change
action_dim = 1      # Dimension of the actions, do not change

actor_nn_params = dict(input_size=state_dim,
                       layers=[(4096, 'relu', 0.9), # (number_of_neurons, activation_function, dropout_rate)
                               (2048, 'relu', 0.9), ],
                       output_size=action_dim,
                       activation_final='tanh')
actor = GenericNeuralNetwork(params=actor_nn_params)

critic_nn_params = dict(input_size=state_dim,
                        layers=[(2048, 'relu', 0.9),
                                (4096, 'relu', 0.9), ],
                        output_size=1,
                        activation_final='linear')
critic = GenericNeuralNetwork(params=critic_nn_params)

param_grid = {
    "gamma": [0.95, 0.97, 0.99],
    "sampling_variance": [0.05, 0.1, 0.2],
    "clip_value": [0.1, 0.2, 0.3],
    "actor_lr": [1e-4, 3e-4, 1e-3],
    "critic_lr": [5e-4, 1e-3, 2e-3],
    "updates_per_iteration": [5, 10],
}

keys, values = zip(*param_grid.items())
combos = random.sample(list(itertools.product(*values)), 30)

results = []
results_per_training_iteration = []
num_iteration = 100
iteration_per_evaluation = 10

for i, combo in enumerate(combos):
    gamma, sampling_var, clip, a_lr, c_lr, updates = combo
    hyper = PPOHyperparameters(
        num_batches=10,
        sampling_variance=sampling_var,
        gamma=gamma,
        updates_per_iteration=updates,
        clip_value=clip,
        actor_lr_start=a_lr,
        actor_lr_end=a_lr,
        critic_lr_start=c_lr,
        critic_lr_end=c_lr,
        cutoff=80,
        use_entropy=False,
        beta_entropy=0.1
    )

    actor = GenericNeuralNetwork(params=actor_nn_params)
    critic = GenericNeuralNetwork(params=critic_nn_params)
    ppo_agent = PPOAgent(environment, state_dim, action_dim, actor, critic, hyper)

    def environment_episode(day_index: int = 0):
        '''
        Reset and interact with the environment until the end of the selected day.

        day_index: Code of the day used in the environment
        '''
        ppo_agent.env.reset(day_index)
        total_reward = 0

        while not environment.done:
            state = ppo_agent.env.state
            action, _ = ppo_agent.sample_action(state, explore=False)
            action = action.detach().item()
            next_state, reward, done, _ = environment.step(ppo_agent.env.descale_action(action))

            total_reward += reward

        return total_reward

    def evaluate_agent():
        evaluation_reward = []
        random_test_days = np.random.choice(test_set_indexes, size=20, replace=False)
        for test_day in random_test_days:
            reward = environment_episode(day_index=test_day)
            evaluation_reward.append(reward)
        return np.mean(evaluation_reward)

    for training_iteration in range(1, num_iteration + 1):
        ppo_agent.train(num_iterations=1)
        if (training_iteration % iteration_per_evaluation) == 0:
            reward = evaluate_agent()   
            print(training_iteration, reward)
            results_per_training_iteration.append({
            "training_iteration": training_iteration, "reward": reward
        })

    results.append({
        "gamma": gamma, "sampling_var": sampling_var, "clip": clip,
        "actor_lr": a_lr, "critic_lr": c_lr, "updates": updates,
        "training_iteration": training_iteration, "reward": reward
    })
    print(f"Run {i+1}/30 → reward {reward:.2f}")

df = pd.DataFrame(results)
df_per_iteration = pd.DataFrame(results_per_training_iteration)
df.to_csv("ppo_results_5.csv", index=False)
df_per_iteration.to_csv("ppo_results_per_iteration_5.csv", index=False)