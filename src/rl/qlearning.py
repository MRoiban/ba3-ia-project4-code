from typing import List, Tuple, Dict, Optional
import numpy as np
import random
from collections import defaultdict
from .env import Labyrinth


class QLearning:
    """
    Q-Learning algorithm for training an agent in a given environment.
    The agent learns an optimal policy for selecting actions to maximize cumulative rewards.

    Attributes:
        env (Labyrinth): The environment in which the agent operates
        gamma (float): Discount factor for future rewards
        alpha (float): Learning rate for Q-value updates
        epsilon (float): Probability of taking a random action (exploration)
        c (float): Parameter for exploration/exploitation balance
        q_values (Dict): Dictionary storing Q-values for state-action pairs
        visit_counts (Dict): Dictionary storing visit counts for state-action pairs
    """

    def __init__(self, env: Labyrinth, gamma: float = 0.9, alpha: float = 0.1, 
                 epsilon: float = 0, c: float = 0):
        """
        Initialize the Q-Learning agent with specified parameters.

        Args:
            env: The environment in which the agent operates
            gamma: Discount factor (0 < gamma <= 1) for future rewards
            alpha: Learning rate (0 < alpha <= 1) for updating Q-values
            epsilon: Probability (0 <= epsilon <= 1) for exploration
            c: Exploration adjustment parameter for UCB-like exploration
        """
        self.env = env
        self.gamma = gamma          
        self.alpha = alpha          
        self.epsilon = epsilon      
        self.c = c                  
        self.q_values: Dict[Tuple[Tuple[int, int], int], float] = defaultdict(float)
        self.visit_counts: Dict[Tuple[Tuple[int, int], int], int] = defaultdict(int)

    def get_q_table(self) -> np.ndarray:
        """
        Retrieve the Q-table as a 3D numpy array for visualization.

        Returns:
            A 3D array representing Q-values for each state-action pair
        """
        height, width = self.env.get_map_size()
        n_actions = len(self.env.get_all_actions())
        q_table = np.zeros((height, width, n_actions))
        
        for state in self.env.get_valid_states() + self.env._world.exit_pos:
            for action in self.env.get_all_actions():
                q_table[state[0], state[1], action] = self.q_values[(state, action)]
                
        return q_table

    def _initialize_tables(self) -> None:
        """
        Initialize Q-values and visit counts tables with zeros.
        """
        valid_states = self.env.get_valid_states() + self.env._world.exit_pos
        for state in valid_states:
            for action in self.env.get_all_actions():
                self.q_values[(state, action)] = 0.0
                self.visit_counts[(state, action)] = 0

    def _select_action_epsilon_greedy(self, current_state: Tuple[int, int]) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            current_state: Current state coordinates

        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.get_all_actions())
        
        return self._get_best_action(current_state)

    def _get_best_action(self, state: Tuple[int, int]) -> int:
        """
        Get the action with the highest Q-value for the given state.

        Args:
            state: Current state coordinates

        Returns:
            Action index with highest Q-value
        """
        best_action = None
        best_q_value = float('-inf')
        for action in self.env.get_all_actions():
            q_value = self.q_values[(state, action)]
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        return best_action

    def _select_action_exploration_bonus(self, current_state: Tuple[int, int]) -> int:
        """
        Select action using Upper Confidence Bound (UCB)-like exploration bonus.

        Args:
            current_state: Current state coordinates

        Returns:
            Selected action index based on Q-values and exploration bonus
        """
        best_action = None
        best_score = float('-inf')

        for action in self.env.get_all_actions():
            q_value = self.q_values[(current_state, action)]
            visits = self.visit_counts[(current_state, action)]
            exploration_bonus = self.c / (visits + 1)
            score = q_value + exploration_bonus
            
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def train(self, n_steps: int) -> List[float]:
        """
        Train the Q-learning agent over a specified number of steps.

        Args:
            n_steps: Total number of steps for training

        Returns:
            List of cumulative rewards for each episode during training
        """
        self._initialize_tables()
        
        current_state = self.env.reset()
        total_steps = 0
        episode_rewards: List[float] = []
        current_episode_reward = 0

        while total_steps < n_steps:
            # Select action based on current policy
            action = (self._select_action_epsilon_greedy(current_state) 
                     if self.epsilon > 0 
                     else self._select_action_exploration_bonus(current_state))

            # Execute action and observe outcome
            reward = self.env.step(action)
            next_state = self.env.get_observation()
            done = self.env.is_done()
            
            current_episode_reward += reward
            self.visit_counts[(current_state, action)] += 1

            # Compute target Q-value
            if done:
                target = reward
            else:
                next_max_q = float('-inf')
                for next_action in self.env.get_all_actions():
                    visits = self.visit_counts[(next_state, next_action)]
                    exploration_bonus = self.c / (visits + 1)
                    q_with_bonus = self.q_values[(next_state, next_action)] + exploration_bonus
                    next_max_q = max(next_max_q, q_with_bonus)
                target = reward + self.gamma * next_max_q

            # Update Q-value
            current_q = self.q_values[(current_state, action)]
            self.q_values[(current_state, action)] = ((1 - self.alpha) * current_q + 
                                                    self.alpha * target)

            # Reset environment if episode is done
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                current_state = self.env.reset()
            else:
                current_state = next_state

            total_steps += 1

        return episode_rewards


