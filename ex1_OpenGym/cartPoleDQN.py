## Template: https://github.com/Amirarsalan-sn/CartPoleV1/
import os
import gc
import torch
import pygame
import warnings
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import math
import minigrid
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Used for debugging; CUDA related errors shown immediately.

# for reproducible results:
seed = 2023
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SarsaMemory:
    def __init__(self, capacity):
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_actions = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, next_state, reward, next_action, done):
        """
        Append (store) the transitions to their respective deques
        """

        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.next_actions.append(next_action)
        self.dones.append(done)

    def data(self):
        """
        return all experience gathered, then convert sampled transitions
        to tensors and move to device (CPU or GPU).
        """

        indices = [i for i in range(len(self))]

        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices]).to(
            device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack(
            [torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        next_actions = torch.as_tensor([self.next_actions[i] for i in indices], dtype=torch.long, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, next_states, rewards, next_actions, dones

    def flush(self):
        self.states.clear()
        self.actions.clear()
        self.next_states.clear()
        self.rewards.clear()
        self.next_actions.clear()
        self.dones.clear()

    def __len__(self):
        """
        To check how many samples are stored in the memory. self.dones deque
        represents the length of the entire memory.
        """

        return len(self.dones)


class ReplayMemory:
    def __init__(self, capacity):
        """
        Experience Replay Memory defined by deques to store transitions/agent experiences
        """

        self.capacity = capacity

        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def store(self, state, action, next_state, reward, done):
        """
        Append (store) the transitions to their respective deques
        """

        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size):
        """
        Randomly sample transitions from memory, then convert sampled transitions
        to tensors and move to device (CPU or GPU).
        """

        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices]).to(
            device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack(
            [torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        """
        To check how many samples are stored in the memory. self.dones deque
        represents the length of the entire memory.
        """

        return len(self.dones)


class DqnNetwork(nn.Module):
    """
    The Deep Q-Network (DQN) model for reinforcement learning.
    This network consists of Fully Connected (FC) layers with ReLU activation functions.
    """

    def __init__(self, num_actions, input_dim):
        """
        Initialize the DQN network.

        Parameters:
            num_actions (int): The number of possible actions in the environment.
            input_dim (int): The dimensionality of the input state space.
        """

        super(DqnNetwork, self).__init__()

        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, num_actions)
        )

        # Initialize FC layer weights using He initialization
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Forward pass of the network to find the Q-values of the actions.

        Parameters:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            Q (torch.Tensor): Tensor containing Q-values for each action.
        """

        Q = self.FC(x)
        return Q


class SarsaAgent:
    def __init__(self, env, epsilon_max, epsilon_min, temp_min, temp, temp_decay, epsilon_decay,
                 epsilon_or_boltzmann: bool, clip_grad_norm, learning_rate, discount, memory_capacity):
        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # RL hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.temp_min = temp_min
        self.temp = temp
        self.temp_decay = temp_decay
        self.epsilon_decay = epsilon_decay
        self.epsilon_or_boltzmann = epsilon_or_boltzmann
        self.discount = discount

        self.action_space = env.action_space
        self.action_space.seed(seed)  # Set the seed to get reproducible results when sampling the action space
        self.observation_space = env.observation_space
        self.sarsa_memory = SarsaMemory(memory_capacity)

        # Initiate the network models
        input_dim = self.observation_space.shape[0]
        output_dim = self.action_space.n

        self.main_network = DqnNetwork(num_actions=output_dim, input_dim=input_dim).to(device)

        self.clip_grad_norm = clip_grad_norm  # For clipping exploding gradients caused by high reward value
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy OR Boltzmann strategy(specified by  self.epsilon_or_boltzmann).

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """
        if self.epsilon_or_boltzmann:
            # Exploration: epsilon-greedy
            if np.random.random() < self.epsilon_max:
                return self.action_space.sample()

            # Exploitation: the action is selected based on the Q-values.
            # Check if the state is a tensor or not. If not, make it a tensor
            if not torch.is_tensor(state):
                state = torch.as_tensor(state, dtype=torch.float32, device=device)

            with torch.no_grad():
                Q_values = self.main_network(state)
                action = torch.argmax(Q_values).item()
        else:  # Exploration: Boltzmann.
            with torch.no_grad():
                if not torch.is_tensor(state):
                    state = torch.as_tensor(state, dtype=torch.float32, device=device)
                q = self.main_network(state)
                q /= self.temp  # dividing each Q(s, a) by the temperature.
                q = torch.nn.functional.softmax(q, dim=0)  # calculating softmax of each Q(s, a)/temp.
                # now, sampling an action using the multinomial distribution calculated above:
                action = torch.multinomial(q, 1).item()

        return action

    def learn(self, done):
        """
        Train the main network using a batch of experiences sampled from the replay memory.

        Parameters:
            done (bool): Indicates whether the episode is done or not. If done,
            calculate the loss of the episode and append it in a list for plot.
        """

        # Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, next_actions, dones = self.sarsa_memory.data()

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        next_actions = next_actions.unsqueeze(1)
        dones = dones.unsqueeze(1)

        #  first calculate the Q target.
        with torch.no_grad():
            next_target_q_value = self.main_network(next_states)
            next_target_q_value = next_target_q_value.gather(dim=1, index=next_actions)

        next_target_q_value[dones] = 0  # Set the Q-value for terminal states to zero
        # forward pass through the main network to find the Q-values of the states:
        predicted_q = self.main_network(states)
        # selecting the Q-values of the actions that were actually taken:
        predicted_q = predicted_q.gather(dim=1, index=actions)

        y_js = rewards + (self.discount * next_target_q_value)  # Compute the target Q-values
        loss = self.criterion(predicted_q, y_js)  # Compute the loss

        # Update the running loss and learned counts for logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            episode_loss = self.running_loss / self.learned_counts  # The average loss for the episode
            self.loss_history.append(episode_loss)  # Append the episode loss to the loss history for plotting
            # Reset the running loss and learned counts
            self.running_loss = 0
            self.learned_counts = 0

        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Perform backward pass and update the gradients

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)

        self.optimizer.step()  # Update the parameters of the main network using the optimizer

    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.

        This method decreases epsilon over time according to a decay factor, ensuring
        that the agent becomes less exploratory and more exploitative as training progresses.
        """

        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def update_boltzmann_temp(self):
        """
        Update the temperature for Boltzmann policy.

        """
        self.temp = max(self.temp_min, self.temp_decay * self.temp_decay)

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.

        """
        torch.save(self.main_network.state_dict(), path)


class DqnAgent:
    """
    DQN Agent Class. This class defines some key elements of the DQN algorithm,
    such as the learning method, hard update, and action selection based on the
    Q-value of actions or the epsilon-greedy policy.
    """

    def __init__(self, env, epsilon_max, epsilon_min, temp_min, temp, temp_decay, epsilon_decay,
                 epsilon_or_boltzmann: bool, clip_grad_norm, learning_rate, discount, memory_capacity):

        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # RL hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.temp_min = temp_min
        self.temp = temp
        self.temp_decay = temp_decay
        self.epsilon_decay = epsilon_decay
        self.epsilon_or_boltzmann = epsilon_or_boltzmann
        self.discount = discount

        self.action_space = env.action_space
        self.action_space.seed(seed)  # Set the seed to get reproducible results when sampling the action space
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity)

        # Initiate the network models
        input_dim = self.observation_space.shape[0]
        output_dim = self.action_space.n

        self.main_network = DqnNetwork(num_actions=output_dim, input_dim=input_dim).to(device)
        self.target_network = DqnNetwork(num_actions=output_dim, input_dim=input_dim).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm  # For clipping exploding gradients caused by high reward value
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy OR Boltzmann strategy(specified by  self.epsilon_or_boltzmann).

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """
        if self.epsilon_or_boltzmann:
            # Exploration: epsilon-greedy
            if np.random.random() < self.epsilon_max:
                return self.action_space.sample()

            # Exploitation: the action is selected based on the Q-values.
            # Check if the state is a tensor or not. If not, make it a tensor
            if not torch.is_tensor(state):
                state = torch.as_tensor(state, dtype=torch.float32, device=device)

            with torch.no_grad():
                Q_values = self.main_network(state)
                action = torch.argmax(Q_values).item()
        else:  # Exploration: Boltzmann.
            if not torch.is_tensor(state):
                state = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                q = self.main_network(state)
                q /= self.temp  # dividing each Q(s, a) by the temperature.
                q = torch.nn.functional.softmax(q, dim=0)  # calculating softmax of each Q(s, a)/temp.
                # now, sampling an action using the multinomial distribution calculated above:
                action = torch.multinomial(q, 1).item()

        return action

    def learn(self, batch_size, done):
        """
        Train the main network using a batch of experiences sampled from the replay memory.

        Parameters:
            batch_size (int): The number of experiences to sample from the replay memory.
            done (bool): Indicates whether the episode is done or not. If done,
            calculate the loss of the episode and append it in a list for plot.
        """

        # Sample a batch of experiences from the replay memory
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # forward pass through the main network to find the Q-values of the states:
        predicted_q = self.main_network(states)
        # selecting the Q-values of the actions that were actually taken:
        predicted_q = predicted_q.gather(dim=1, index=actions)

        # Compute the maximum Q-value for the next states using the target network
        with torch.no_grad():
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[
                0]  # not argmax (cause we want the maxmimum q-value, not the action that maximize it)

        next_target_q_value[dones] = 0  # Set the Q-value for terminal states to zero
        y_js = rewards + (self.discount * next_target_q_value)  # Compute the target Q-values
        loss = self.critertion(predicted_q, y_js)  # Compute the loss

        # Update the running loss and learned counts for logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            episode_loss = self.running_loss / self.learned_counts  # The average loss for the episode
            self.loss_history.append(episode_loss)  # Append the episode loss to the loss history for plotting
            # Reset the running loss and learned counts
            self.running_loss = 0
            self.learned_counts = 0

        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Perform backward pass and update the gradients

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)

        self.optimizer.step()  # Update the parameters of the main network using the optimizer

    def hard_update(self):
        """
        Navie update: Update the target network parameters by directly copying
        the parameters from the main network.
        """

        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.

        This method decreases epsilon over time according to a decay factor, ensuring
        that the agent becomes less exploratory and more exploitative as training progresses.
        """

        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def update_boltzmann_temp(self):
        """
        Update the temperature for Boltzmann policy.

        """
        self.temp = max(self.temp_min, self.temp_decay * self.temp)

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.

        """
        torch.save(self.main_network.state_dict(), path)


class StepWrapper(gym.Wrapper):
    """
    A wrapper class for modifying the state and reward functions of the
    MountainCar-v0 environment.
    """

    def __init__(self, env):
        """
        Initializes the StepWrapper. This is the main class for wrapping the environment with it.

        Args:
            env (gym.Env): The environment to be wrapped.

        Attributes:
            reward_wrapper (Inherit from RewardWrapper):
                An instance of the RewardWrapper class for modifying rewards.
        """
        super().__init__(env)  # We give the env here to initialize the gym.Wrapper superclass (inherited).
        self.observation_wrapper = ObservationWrapper(env)
        self.reward_wrapper = RewardWrapper(env)

    def step(self, action):
        """
        Executes a step in the environment with the provided action.The reason
        behind using this method is to have access to the state and reward functions return.

        Args:
            action (int): The action to be taken.
        """

        state, reward, done, truncation, info = self.env.step(action)  # Same as before as usual

        modified_reward = self.reward_wrapper.reward(
            state)  # Give the modified state to another Wrapper to return the modified reward
        modified_state = self.observation_wrapper.observation(state)
        return state, modified_reward, done, truncation, info  # The same returns as usual but with modified versions of the state and reward functions

    def reset(self, seed):
        state, info = self.env.reset(seed=seed)  # Same as before as usual
        modified_state = self.observation_wrapper.observation(state)
        return state, info  # Same as before as usual but with returning the modified version of the state


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, state):  # state normalizer
        state = np.array(state)
        state[0] = (state[0] + 4.8) / 9.6
        state[1] = (state[1] + 10) / 20
        state[2] = (state[2] + 0.418) / 0.836
        state[3] = (state[3] + 10) / 20

        return state


class RewardWrapper(gym.RewardWrapper):
    """
    Wrapper class for modifying rewards in the MountainCar-v0 environment.

    Args:
        env (gym.Env): The environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, state):
        """
        Modifies the reward based on the current state of the environment.

        Args:
            state (numpy.ndarray): The current state of the environment.

        Returns:
            float: The modified reward.
        """
        current_position = state[0]
        current_angel = state[2]

        position_reward = math.cos((2 * math.pi * current_position) / 4.8)
        angel_reward = math.cos((2 * math.pi * current_angel) / 0.419)

        modified_reward = position_reward + 2 * angel_reward
        return modified_reward


class ModelTrainTest():
    def __init__(self, hyperparams):

        # Define RL Hyperparameters
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]

        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]

        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.temp_min = hyperparams["temp_min"]
        self.temp = hyperparams["temp"]
        self.temp_decay = hyperparams["temp_decay"]
        self.epsilon_or_boltzmann = hyperparams["epsilon_or_boltzmann"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        self.memory_capacity = hyperparams["memory_capacity"]

        self.dqn_or_sarsa = hyperparams["dqn_or_sarsa"]

        self.render_fps = hyperparams["render_fps"]

        # Define Env
        self.env = gym.make('CartPole-v1', max_episode_steps=self.max_steps,
                            render_mode="human" if self.render else None)
        self.env.metadata['render_fps'] = self.render_fps  # For max frame rate make it 0

        """
        The 'CartPole-v1' environment in the 'gymnasium' library generates 
        UserWarnings about deprecated methods.
        These warnings are related to the 'size' and 'shape' methods which are 
        being phased out in a future version of the library.
        Even though we are not directly using these methods in our code, the 
        warnings are still displayed.
        To keep our output clean and focused on our own program's execution, 
        the following line of code is for ignoring these warnings.
        """
        warnings.filterwarnings("ignore", category=UserWarning)

        # Apply RewardWrapper
        self.env = StepWrapper(self.env)
        if self.dqn_or_sarsa:
            # Define the agent class
            self.agent = DqnAgent(env=self.env,
                                  epsilon_max=self.epsilon_max,
                                  epsilon_min=self.epsilon_min,
                                  temp_min=self.temp_min,
                                  temp=self.temp,
                                  temp_decay=self.temp_decay,
                                  epsilon_or_boltzmann=self.epsilon_or_boltzmann,
                                  epsilon_decay=self.epsilon_decay,
                                  clip_grad_norm=self.clip_grad_norm,
                                  learning_rate=self.learning_rate,
                                  discount=self.discount_factor,
                                  memory_capacity=self.memory_capacity)
        else:
            self.agent = SarsaAgent(env=self.env,
                                    epsilon_max=self.epsilon_max,
                                    epsilon_min=self.epsilon_min,
                                    temp_min=self.temp_min,
                                    temp=self.temp,
                                    temp_decay=self.temp_decay,
                                    epsilon_or_boltzmann=self.epsilon_or_boltzmann,
                                    epsilon_decay=self.epsilon_decay,
                                    clip_grad_norm=self.clip_grad_norm,
                                    learning_rate=self.learning_rate,
                                    discount=self.discount_factor,
                                    memory_capacity=self.memory_capacity)

    def dqn_train(self):
        total_steps = 0
        self.reward_history = []

        # Training loop over episodes
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                self.agent.replay_memory.store(state, action, next_state, reward, done)

                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))

                    # Update target-network weights
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()

                state = next_state
                episode_reward += reward
                step_size += 1

            # Appends for tracking history
            self.reward_history.append(episode_reward)  # episode reward
            total_steps += step_size

            # Decay epsilon at the end of each episode
            if self.epsilon_or_boltzmann:
                self.agent.update_epsilon()
            else:
                self.agent.update_boltzmann_temp()

            # -- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')
            if self.epsilon_or_boltzmann:
                result = (f"Episode: {episode}, " /
                          f"Total Steps: {total_steps}, " /
                          f"Ep Step: {step_size}, " /
                          f"Raw Reward: {episode_reward:.2f}, " /
                          f"Epsilon: {self.agent.epsilon_max:.2f}")
            else:
                result = (f"Episode: {episode}, " /
                          f"Total Steps: {total_steps}, " /
                          f"Ep Step: {step_size}, " /
                          f"Raw Reward: {episode_reward:.2f}, " /
                          f"Boltzmann: {self.agent.temp:.2f}")
            print(result)
        self.plot_training(episode)

    def sarsa_train(self):
        total_steps = 0
        self.reward_history = []

        # Training loop over episodes
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0
            exp_iter = 1
            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)
                next_action = self.agent.select_action(next_state)
                self.agent.sarsa_memory.store(state, action, next_state, reward, next_action, done)

                if exp_iter == self.memory_capacity or (done or truncation):
                    self.agent.learn((done or truncation))
                    exp_iter = 0
                    self.agent.sarsa_memory.flush()  # it's necessary to flush memory because the policy has changed.

                state = next_state
                episode_reward += reward
                step_size += 1
                exp_iter += 1

            # Appends for tracking history
            self.reward_history.append(episode_reward)  # episode reward
            total_steps += step_size

            # Decay epsilon at the end of each episode
            if self.epsilon_or_boltzmann:
                self.agent.update_epsilon()
            else:
                self.agent.update_boltzmann_temp()

            # -- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')
            if self.epsilon_or_boltzmann:
                result = (f"Episode: {episode}, "
                          f"Total Steps: {total_steps}, " /
                          f"Ep Step: {step_size}, "/
                          f"Raw Reward: {episode_reward:.2f}, "/
                          f"Epsilon: {self.agent.epsilon_max:.2f}")
            else:
                result = (f"Episode: {episode}, "
                          f"Total Steps: {total_steps}, "/
                          f"Ep Step: {step_size}, "/
                          f"Raw Reward: {episode_reward:.2f}, "/
                          f"Boltzmann: {self.agent.temp:.2f}")
            print(result)
        self.plot_training(episode)

    def train(self):
        """
        Reinforcement learning training loop.
        """
        if self.dqn_or_sarsa:
            self.dqn_train()
        else:
            self.sarsa_train()

    def test(self, max_episodes):
        """
        Reinforcement learning policy evaluation.
        """

        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()

        # Testing loop over episodes
        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                state = next_state
                episode_reward += reward
                step_size += 1

            # Print log
            result = (f"Episode: {episode}, " /
                      f"Steps: {step_size:}, " /
                      f"Reward: {episode_reward:.2f}, ")
            print(result)

        pygame.quit()  # close the rendering window

    def plot_training(self, episode):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(self.reward_history, np.ones(50) / 50, mode='valid')

        # Clip max (high) values for better plot analysis
        reward_history = np.clip(self.reward_history, a_min=None, a_max=100)
        sma = np.clip(sma, a_min=None, a_max=100)

        plt.figure()
        plt.title("Obtained Rewards")
        plt.plot(reward_history, label='Raw Reward', color='#4BA754', alpha=1)
        plt.plot(sma, label='SMA 50', color='#F08100')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.title("Network Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#8921BB', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    """
    DQN epsilon-greedy -> 1475.94 : Pure rewards from one episode.
    DQN boltzmann -> 1362.23
    SARSA epsilon-greedy -> 1484.33
    """
    # Parameters:
    train_mode = False
    render = not train_mode
    RL_hyperparams = {
        "train_mode": train_mode,
        "RL_load_path": './dqn_epsilon/final_weights' + '_' + '900' + '.pth',
        "save_path": './dqn_epsilon/final_weights',
        "save_interval": 100,

        "clip_grad_norm": 5,
        "learning_rate": 75e-5,
        "discount_factor": 0.96,
        "batch_size": 64,
        "update_frequency": 20,
        "max_episodes": 1000 if train_mode else 2,
        "max_steps": 500,
        "render": render,

        "epsilon_max": 0.999 if train_mode else -1,
        "epsilon_min": 0.01,
        "temp_min": 0.1,
        "temp": 15 if train_mode else 0.1,
        "temp_decay": 0.994,
        "epsilon_or_boltzmann": True,
        "epsilon_decay": 0.997,
        "dqn_or_sarsa": True,
        "memory_capacity": 125_000 if train_mode else 0,

        "render_fps": 60,
    }

    # Run
    DRL = ModelTrainTest(RL_hyperparams)  # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes=RL_hyperparams['max_episodes'])