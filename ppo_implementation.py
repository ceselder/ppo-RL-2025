from Environment import BaseEnvironment
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
import numpy as np
from nn_utils import GenericNeuralNetwork
import pandas as pd, matplotlib.pyplot as plt
import pickle

#hyperparameters I declared to make stuff go faster locally
big_layer = 512 # default = 4096
smaller_layer = 256 #default = 2048

with open('public_data_dict.pkl', 'rb') as f:
    data = pickle.load(f)

class PPOHyperparameters():

    def __init__(self, num_batches: int, sampling_variance: float, gamma: float, updates_per_iteration: int, clip_value: float, actor_lr: float, critic_lr: float, ):
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
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

class PPOAgent:

    def __init__(self, env, state_dim, action_dim, actor: nn.Module, critic: nn.Module, hyperparameters: PPOHyperparameters):

        self.env = env

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hyperparameters = hyperparameters

        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hyperparameters.actor_lr)

        self.critic = critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hyperparameters.critic_lr)

        self.cov_var = torch.full(size=(self.action_dim,), fill_value = self.hyperparameters.sampling_variance)
        self.cov_mat = torch.diag(self.cov_var) # This will be used to sample actions from the agent distribution. The higher the variance, the wider the gaussian distribution.


    def train(self, num_iterations:int = 200):
        for iteration in range(num_iterations):
            batch_states, batch_actions, batch_log_probs, batch_cumulative_rewards, batch_rewards = self.rollout()

            V, _ = self.calculate_value_logprobs(batch_states=batch_states, batch_actions=batch_actions)
            A_k = batch_cumulative_rewards - V.detach().squeeze(1)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)

            for _ in range(self.hyperparameters.updates_per_iteration):
                current_V, current_log_probs = self.calculate_value_logprobs(batch_states=batch_states, batch_actions=batch_actions)
                current_V = current_V.squeeze(1)
                probability_ratios = torch.exp(current_log_probs - batch_log_probs)
                actor_loss = -torch.min(probability_ratios * A_k, torch.clamp(probability_ratios, 1 - self.hyperparameters.clip_value, 1 + self.hyperparameters.clip_value) * A_k).mean()
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
        log_probs = dist.log_prob(batch_actions)
        return V, log_probs

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
        return batch_cumulative_rewards

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
                state, reward, done, _ = self.env.step(action.item())
                rewards_episode.append(reward)

            batch_rewards.append(rewards_episode)

        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float).unsqueeze(dim=-1)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float)

        batch_cumulative_rewards = self.compute_cumulative_rewards(batch_rewards=batch_rewards)     # (timesteps_per_episode)

        return batch_states, batch_actions, batch_log_probs, batch_cumulative_rewards, batch_rewards

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
                       layers=[(big_layer, 'relu', 0.9), # (number_of_neurons, activation_function, dropout_rate)
                               (smaller_layer, 'relu', 0.9), ],
                       output_size=action_dim,
                       activation_final='tanh')
actor = GenericNeuralNetwork(params=actor_nn_params)

critic_nn_params = dict(input_size=state_dim,
                        layers=[(big_layer, 'relu', 0.9),
                                (smaller_layer, 'relu', 0.9), ],
                        output_size=1,
                        activation_final='linear')
critic = GenericNeuralNetwork(params=critic_nn_params)

hyperparameters = PPOHyperparameters(num_batches=10,
                                    sampling_variance=0.1,
                                    gamma=0.99,
                                    updates_per_iteration=10,
                                    clip_value=0.2,
                                    actor_lr=3e-4,
                                    critic_lr=1e-3)

ppo_agent = PPOAgent(env=environment, state_dim=state_dim, action_dim=action_dim, actor=actor, critic=critic, hyperparameters=hyperparameters)

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

num_iteration = 100
iteration_per_evaluation = 10

for training_iteration in range(1, num_iteration+1):
    ppo_agent.train(num_iterations=1)

    if (training_iteration % iteration_per_evaluation) == 0:
        reward = evaluate_agent()
        print(training_iteration, reward)

summary = pd.read_csv("ppo_results.csv")
curves  = pd.read_csv("ppo_results_per_iteration.csv")

assert len(curves) % len(summary) == 0, "Curves length not multiple of runs!"

steps_per_run = len(curves) // len(summary)

for i, row in summary.reset_index(drop=True).iterrows():
    gamma        = row["gamma"]
    sampling_var = row["sampling_var"]
    clip         = row["clip"]
    actor_lr     = row["actor_lr"]
    critic_lr    = row["critic_lr"]
    updates      = row["updates"]

    start = i * steps_per_run
    end   = start + steps_per_run
    run_curve = curves.iloc[start:end]

    print(f"\nRun {i+1}/{len(summary)}")
    print(f"  γ = {gamma} | σ = {sampling_var} | clip = {clip} | "
          f"actor lr = {actor_lr} | critic lr = {critic_lr} | updates = {updates}")

    plt.figure(figsize=(6,4))
    plt.plot(run_curve["training_iteration"], run_curve["reward"], marker="o")
    plt.title(f"Reward evolution — γ={gamma}, σ={sampling_var}, clip={clip}, upd={updates}")
    plt.xlabel("Training iteration")
    plt.ylabel("Mean evaluation reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()