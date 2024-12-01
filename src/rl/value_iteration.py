import numpy as np
from .env import Labyrinth
from collections import defaultdict
import random
from typing import List, Tuple, Dict, Optional


class ValueIteration:
    """
    Value Iteration algorithm for solving a reinforcement learning environment.
    The algorithm iteratively updates the estimated values of states to find an optimal policy.

    Attributes:
        env (Labyrinth): The environment in which the agent operates
        gamma (float): Discount factor for future rewards
        theta (float): Convergence threshold for value iteration
        env_size (Tuple[int, int]): Size of the environment grid
        value_table (np.ndarray): Table storing state values
        rewards (Dict): Dictionary storing reward values for state-action-state transitions
        transitions (Dict): Dictionary storing transition probabilities
    """
    
    def __init__(self, env: Labyrinth, gamma: float = 1):
        """
        Initialize the Value Iteration agent with specified parameters.

        Args:
            env: The environment in which the agent operates
            gamma: Discount factor (0 < gamma <= 1) for future rewards
        """
        self.env = env
        self.gamma = gamma  
        self.theta = 0.01
        self.env_size = env.get_map_size()
        self.value_table = np.zeros(self.env_size)
        self.rewards: Dict = {}
        self.transitions: Dict = {}

    def transition(self, state: Tuple[int, int], action: int, s_prime: Tuple[int, int]) -> float:
        """
        Get the transition probability from state to s_prime under action.

        Args:
            state: Current state coordinates
            action: Action to take
            s_prime: Next state coordinates

        Returns:
            Probability of transitioning from state to s_prime under action
        """
        return self.transitions[state][action][s_prime]

    def step(self, action: int, state: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take a step in the environment with the given action.

        Args:
            action: Action to take in the environment
            state: Optional state to set before taking action

        Returns:
            Tuple containing (next_state, reward, is_done)
        """
        if state is not None:
            self.env.set_state(state)
        reward = self.env.step(action)
        is_done = self.env.is_done()
        return self.env.get_observation(), reward, is_done

    def random_strategy(self, _state: Tuple[int, int]) -> int:
        """
        Return a random action from the available actions.

        Args:
            _state: Current state (unused in random strategy)

        Returns:
            Random action index
        """
        return random.choice(self.env.get_all_actions())

    def collect_episodes(self, n_samples: int, n_steps: int, exploration_strategy) -> List[List[Tuple]]:
        """
        Collect episodes using the specified exploration strategy.

        Args:
            n_samples: Number of episodes to collect
            n_steps: Maximum number of steps per episode
            exploration_strategy: Function that returns actions given states

        Returns:
            List of episodes, where each episode is a list of (state, action, reward, next_state) tuples
        """
        episodes = []
        for _ in range(n_samples):
            episode = []
            state = self.env.reset()
            for _ in range(n_steps):
                action = exploration_strategy(state)
                s_prime, reward, done = self.step(action)
                episode.append((state, action, reward, s_prime))
                state = s_prime
                if done:
                    break
            episodes.append(episode)
        return episodes

    def learn(self, n_samples: int, n_steps: int) -> None:
        """
        Learn the transition and reward models from collected episodes.

        Args:
            n_samples: Number of episodes to collect
            n_steps: Maximum number of steps per episode
        """
        episodes = self.collect_episodes(n_samples, n_steps, self.random_strategy)
        states = self.env.get_valid_states() + self.env._world.exit_pos
        actions = self.env.get_all_actions()

        transition_counts = {s: {a: {s_prime: 0 for s_prime in states} for a in actions} for s in states}
        reward_sums = {s: {a: {s_prime: 0 for s_prime in states} for a in actions} for s in states}
        
        for episode in episodes:
            for (s, a, r, s_prime) in episode:
                if s_prime in states:
                    transition_counts[s][a][s_prime] += 1
                    reward_sums[s][a][s_prime] += r
        
        self.transitions = {
            s: {
                a: {
                    s_prime: (transition_counts[s][a][s_prime] / sum(transition_counts[s][a].values()))
                    if sum(transition_counts[s][a].values()) > 0 else 0
                    for s_prime in states
                }
                for a in actions
            }
            for s in states
        }
        self.rewards = {
            s: {
                a: {
                    s_prime: (reward_sums[s][a][s_prime] / transition_counts[s][a][s_prime])
                    if transition_counts[s][a][s_prime] > 0 else 0
                    for s_prime in states
                }
                for a in actions
            }
            for s in states
        }

    def bellman_equation(self, state: Tuple[int, int]) -> float:
        """
        Compute the Bellman equation for a given state.

        Args:
            state: The state to compute the Bellman equation for

        Returns:
            Maximum value over all actions for the given state
        """
        actions = self.env.get_all_actions()
        values = np.zeros(len(actions))
        valid_states = self.env.get_valid_states() + self.env._world.exit_pos
        
        for action in actions:
            values[action] = sum(
                self.transitions[state][action][s_prime] * 
                (self.rewards[state][action][s_prime] + (self.gamma * self.value_table[s_prime]))
                for s_prime in valid_states
            )
        return max(values)

    def train(self, n_updates: int) -> None:
        """
        Train the agent using value iteration for a specified number of updates.

        Args:
            n_updates: The total number of updates to perform
        """
        self.learn(100_000, 2_000)
        delta = 0
        for _ in range(n_updates):
            delta = self.value_iteration(delta)
            if delta <= self.theta:
                break

    def value_iteration(self, delta: float) -> float:
        """
        Perform one iteration of the value iteration algorithm.

        Args:
            delta: Current maximum change in value estimates

        Returns:
            Updated maximum change in value estimates
        """
        valid_states = self.env.get_valid_states() + self.env._world.exit_pos
        for state in valid_states:
            value = self.value_table[state]
            self.value_table[state] = self.bellman_equation(state)
            delta = max(delta, abs(value - self.value_table[state]))
        return delta

    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
            A 2D array representing the estimated values for each state
        """
        return self.value_table
