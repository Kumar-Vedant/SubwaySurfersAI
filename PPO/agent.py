import torch
import numpy as np
from neural_net import AgentNN


class Agent:
    def __init__(self, input_dims, num_actions, lr=1e-3, gamma=0.95, lam=0.95, policy_clip=0.2, value_coef=0.5, entropy_coef=0.01, obs_size=2048, batch_size=64, n_epochs=8):

        self.num_actions = num_actions
        self.learn_step_counter = 0

        # set hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.policy_clip = policy_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.obs_size = obs_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # create the network that has both actor and critic
        self.network = AgentNN(input_dims, num_actions)

        # set optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        # setup observations memory (state, action, reward, action_log_prob, value, done)
        self.observations = [[], [], [], [], [], []]

    # pass the state to actor and critic networks to choose an action
    def choose_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.network.device)
        
        # get the probability distribution over actions and the value of this state
        dist, value = self.network(state)

        # get an action by sampling the probability distribution and calculate the log-probability of the action taken
        action = dist.sample()

        prob = torch.squeeze(dist.log_prob(action)).item()
        # prob = dist.log_prob(action).item()
        action = torch.squeeze(action).item()

        value = torch.squeeze(value).item()

        return action, prob, value

    # store a 5-tuple in the replay buffer (state, action, reward, action_log_prob, value) along with a "done" flag
    def store_in_memory(self, state, action, reward, action_log_prob, value, done):
        self.observations[0].append(np.array(state))
        self.observations[1].append(action)
        self.observations[2].append(reward)
        self.observations[3].append(action_log_prob)
        self.observations[4].append(value)
        self.observations[5].append(done)

    # calculate advantages using General Advantage Estimation (GAEs)
    def calculate_advantage(self, rewards, values, dones):
        rewards = torch.tensor(rewards).to(self.network.device)

        advantages = torch.zeros_like(rewards).to(self.network.device)
        gae = 0

        # calculate using bootstrapping
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(torch.load(path))

    def learn(self, next_state):
        # if there aren't enough observations to complete a batch
        if len(self.observations[0]) < self.obs_size:
            return
        
        # extract all information required from observations
        states, actions, rewards, old_log_probs, values, dones = self.observations

        # convert to tensorflow format
        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.network.device)
        actions = torch.tensor(actions).to(self.network.device)
        old_log_probs = torch.tensor(old_log_probs).to(self.network.device)
        values = torch.tensor(values).to(self.network.device)

        # calculate advantages and returns
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0).to(self.network.device)
        _, value = self.network(next_state)
        value = value.squeeze().detach()
        values = torch.cat([values, value.unsqueeze(0)], dim=0)

        advantages = self.calculate_advantage(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + values[:-1]

        indices = np.arange(self.obs_size)

        for _ in range(self.n_epochs):
            # randomize indices
            np.random.shuffle(indices)

            for start in range(0, self.obs_size, self.batch_size):
                # get the indices for this mini-batch
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # pass the states back into the network to get current policy and values for updation
                dists, critic_values = self.network(mb_states)

                new_log_probs = dists.log_prob(mb_actions)
                entropy = dists.entropy().mean()

                # calculate actor loss using clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                weighted_probs = mb_advantages * ratio
                weighted_clipped_probs = torch.clamp(ratio, 1-self.policy_clip, 1+self.policy_clip) * mb_advantages

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # calculate critic loss using MSE
                critic_loss = (critic_values-mb_returns)**2
                critic_loss = critic_loss.mean()

                # compute total loss with entropy regularization to encourage exploration
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # reset gradients to avoid accumulation from previous steps
                self.optimizer.zero_grad()

                # calculate gradients using backpropagation
                loss.backward()
                # perform gradient descent using those gradients
                self.optimizer.step()

        self.learn_step_counter += 1
        # reset observation memory
        self.observations = [[], [], [], [], [], []]