import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Buffer:
    def __init__(self, args, device):
        self.buffer_capacity = args.buffer_capacity
        self.batch_size = args.batch_size
        self.buffer_counter = 0
        self.device = device

        # memory buffers
        self.state_buffer = np.zeros((self.buffer_capacity, args.state_size, args.state_size), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, args.num_actions), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, args.state_size, args.state_size), dtype=np.float32)

        # optimizers will be passed from training loop
        self.actor_optimizer = None
        self.critic_optimizer = None

    def record(self, obs_tuple):
        """Record (state, action, reward, next_state) tuple in the buffer."""
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
               actor_model, critic_model, target_actor, target_critic, gamma,
               actor_optimizer, critic_optimizer):
        """Perform one gradient update step for actor and critic networks."""

        # Convert batches to tensors on the correct device
        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device).unsqueeze(1)   # [B,1,H,W]
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        action_batch = torch.tensor(action_batch, dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)

        # ------------------ CRITIC UPDATE ------------------
        with torch.no_grad():
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic(next_state_batch, target_actions)

        critic_value = critic_model(state_batch, action_batch)
        critic_loss = nn.MSELoss()(critic_value, y)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # ------------------ ACTOR UPDATE ------------------
        actions_pred = actor_model(state_batch)
        actor_loss = -critic_model(state_batch, actions_pred).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

    def learn(self, actor_model, critic_model, target_actor, target_critic,
              actor_optimizer, critic_optimizer, args):
        """Sample a batch and perform one learning step."""
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]

        self.update(state_batch, action_batch, reward_batch, next_state_batch,
                    actor_model, critic_model, target_actor, target_critic,
                    args.gamma, actor_optimizer, critic_optimizer)
