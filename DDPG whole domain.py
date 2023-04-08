import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 512)
        self.layer_2 = nn.Linear(512, 512)
        self.layer_3 = nn.Linear(512, 256)
        self.layer_4 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = nn.ReLU()(self.layer_1(x))
        x = nn.ReLU()(self.layer_2(x))
        x = nn.ReLU()(self.layer_3(x))
        x = self.max_action * nn.Tanh()(self.layer_4(x))
        return x

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 512)
        self.layer_2 = nn.Linear(512, 512)
        self.layer_3 = nn.Linear(512, 256)
        self.layer_4 = nn.Linear(256, 1)

    def forward(self, x, u):
        x = nn.ReLU()(self.layer_1(torch.cat([x, u], 1)))
        x = nn.ReLU()(self.layer_2(x))
        x = nn.ReLU()(self.layer_3(x))
        x = self.layer_4(x)
        return x

class DDPG():
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.memory = deque(maxlen=100000)
        self.batch_size = 100
        self.discount = 0.9
        self.tau = 0.001
        self.max_action = max_action

    def add_to_memory(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state, action, next_state, reward, done = zip(*batch)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        # Train the Critic
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * self.discount * target_Q
        current_Q = self.critic(state, action)

        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train the Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()


def noisy_gaussian(state):
    x, y, z = state
    SNR = 0.1
    mu_x, mu_y, mu_z = 0, 0, 0
    sigma_x, sigma_y, sigma_z = 1, 1, 1
    f = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2) + (z - mu_z) ** 2 / (2 * sigma_z ** 2)))
    noisy_f = f + np.random.normal(0, 1 / SNR, f.shape)
    return noisy_f

#DDPG settings
state_dim = 3
action_dim = 4 # 3 dimensions for the searching center and 1 dimension for the radius
max_action = 2 # The maximum value of each component of the action

#Initialize DDPG agent
agent = DDPG(state_dim, action_dim, max_action)

n_episodes = 2
max_steps = 50

#Exploration noise
exploration_noise = 0.01

max_reward = float('-inf')
max_state = None

for episode in range(n_episodes):
    state = np.random.uniform(-1, 1, state_dim)
    episode_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        action = (action + exploration_noise * np.random.randn(action_dim)).clip(-max_action, max_action)
        
        # Sample a point within the sphere defined by the center and radius given by the action
        center, radius = action[:3], action[3]
        random_direction = np.random.randn(3)
        random_direction /= np.linalg.norm(random_direction)
        random_distance = radius * np.random.random() ** (1/3)
        next_state = center + random_distance * random_direction

        reward_factor = 100
        reward = reward_factor * noisy_gaussian(next_state)
        done = False

        agent.add_to_memory(state, action, next_state, reward, done)
        agent.train()

        state = next_state
        episode_reward += reward

        if reward > max_reward:
            max_reward = reward
            max_state = next_state
            distance = np.sqrt(max_state[0]**2 + max_state[1]**2 + max_state[2]**2)

        if step == max_steps - 1:
            print("Episode", episode, "Reward", episode_reward/reward_factor)
            print(state,np.sqrt(state[0]**2 + state[1]**2 + state[2]**2)<0.225)
            break
print("Maximum location:", max_state, max_reward/reward_factor)
distance = np.sqrt(max_state[0]**2 + max_state[1]**2 + max_state[2]**2)
success = distance <0.224
print(success,distance)
