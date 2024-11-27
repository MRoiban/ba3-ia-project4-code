from .env import Labyrinth

import numpy as np
import random
from collections import defaultdict

class QLearning:
    """
    Q-Learning algorithm for training an agent in a given environment.
    The agent learns an optimal policy for selecting actions to maximize cumulative rewards.

    Attributes:
    - env (Labyrinth): The environment in which the agent operates.
    - gamma (float): Discount factor for future rewards.
    - alpha (float): Learning rate.
    - epsilon (float): Probability of taking a random action (exploration).
    - c (float): Parameter for exploration/exploitation balance in action selection.
    ...
    """

    def __init__(self, env: Labyrinth, gamma: float = 0.9, alpha: float = 0.1, epsilon: float = 0, c: float = 0):
        """
        Initialize the Q-Learning agent with specified parameters.

        Parameters:
        - env (Labyrinth): The environment in which the agent operates.
        - gamma (float): Discount factor (0 < gamma <= 1) for future rewards.
        - alpha (float): Learning rate (0 < alpha <= 1) for updating Q-values.
        - epsilon (float): Probability (0 <= epsilon <= 1) for exploration in action selection.
        - c (float): Exploration adjustment parameter.
        """
        self.env = env
        self.gamma = gamma          
        self.alpha = alpha          
        self.epsilon = epsilon      
        self.c = c                  
        self.q_values = defaultdict(float)  # Ensure default values are zero
        self.visit_counts = defaultdict(int)  # Ensure default counts are zero

    def get_q_table(self) -> np.ndarray:
        """
        Retrieve the Q-table as a 3D numpy array for visualization.

        Returns:
        - np.ndarray: A 3D array representing Q-values for each state-action pair.
        """
        height, width = self.env.get_map_size()
        n_actions = len(self.env.get_all_actions())
        q_table = np.zeros((height, width, n_actions))
        
        for state in self.env.get_valid_states():
            for action in self.env.get_all_actions():
                q_table[state[0], state[1], action] = self.q_values[(state, action)]
                
        return q_table

    def _initialize_tables(self):
        """
        Initialize Q-values and visit counts tables.
        """
        for state in self.env.get_valid_states():
            for action in self.env.get_all_actions():
                self.q_values[(state, action)] = 0.0
                self.visit_counts[(state, action)] = 0

    def _select_action_epsilon_greedy(self, current_state):
        """
        Select action using epsilon-greedy policy.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.get_all_actions())
        return max(self.env.get_all_actions(), 
                  key=lambda a: self.q_values[(current_state, a)])

    def _select_action_exploration_bonus(self, current_state):
        """
        Select action using exploration bonus policy.
        Adheres to the specified formula for action selection.
        """
        best_action = None
        best_score = float('-inf')

        for action in self.env.get_all_actions():
            q_value = self.q_values[(current_state, action)]
            visits = self.visit_counts[(current_state, action)]
            exploration_bonus = self.c / (visits + 1)  # Exploration bonus formula
            score = q_value + exploration_bonus
            
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def train(self, n_steps: int):
        """
        Train the Q-learning agent over a specified number of steps.

        Parameters:
        - n_steps (int): Total number of steps for training.
        
        Returns:
        - list: List of cumulative rewards for each episode during training.
        """
        self._initialize_tables()
        
        current_state = self.env.reset()
        total_steps = 0
        episode_rewards = []
        current_episode_reward = 0

        while total_steps < n_steps:
            if self.epsilon > 0:
                action = self._select_action_epsilon_greedy(current_state)
            else:
                action = self._select_action_exploration_bonus(current_state)

            # Take action and observe reward and next state
            reward = self.env.step(action)
            next_state = self.env.get_observation()
            done = self.env.is_done()
            
            # Track episode reward
            current_episode_reward += reward

            self.visit_counts[(current_state, action)] += 1

            # Q-value update logic
            if done:
                target = reward
            else:
                next_max_q = float('-inf')
                for next_action in self.env.get_all_actions():
                    visits = self.visit_counts[(next_state, next_action)]
                    exploration_bonus = self.c / (visits + 1)  # Exploration bonus for next state-action pair
                    q_with_bonus = self.q_values[(next_state, next_action)] + exploration_bonus
                    next_max_q = max(next_max_q, q_with_bonus)

                target = reward + self.gamma * next_max_q

            # Update Q-value
            current_q = self.q_values[(current_state, action)]
            self.q_values[(current_state, action)] = (1 - self.alpha) * current_q + \
                                                    self.alpha * target

            # Reset if done, otherwise continue
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                current_state = self.env.reset()
            else:
                current_state = next_state

            total_steps += 1

        return episode_rewards


def train_multiple_agents(env, n_agents=20, n_steps=10000, **agent_params):
    """
    Train multiple Q-learning agents and collect their rewards.

    Parameters:
    - env: Environment instance
    - n_agents (int): Number of agents to train.
    - n_steps (int): Number of steps per agent.
    - **agent_params: Parameters for the QLearning agent.

    Returns:
    - list: A list containing cumulative rewards for each episode for all agents.
    """
    all_episode_rewards = []
    for _ in range(n_agents):
        agent = QLearning(env, **agent_params)
        episode_rewards = agent.train(n_steps)
        all_episode_rewards.append(episode_rewards)
    
    return all_episode_rewards