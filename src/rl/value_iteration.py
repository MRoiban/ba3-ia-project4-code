import lle
import numpy as np
from .env import Labyrinth
from collections import defaultdict
import random
class ValueIteration:
    """
    Value Iteration algorithm for solving a reinforcement learning environment.
    The algorithm iteratively updates the estimated values of states to find an optimal policy.

    Attributes:
    - env (Labyrinth): The environment in which the agent operates.
    - gamma (float): Discount factor for future rewards.
    ...
    """
    
    def __init__(self, env: Labyrinth, gamma: float = 1):
        """
        Initialize the Value Iteration agent with specified parameters.

        Parameters:
        - env (Labyrinth): The environment in which the agent operates.
        - gamma (float): Discount factor (0 < gamma <= 1) for future rewards.
        """
        self.env = env
        self.gamma = gamma  
        self.theta = 0.01
        self.env_size = env.get_map_size()
        self.value_table = np.zeros(self.env_size)
        self.rewards = {}
        self.transitions = {}


    def transition(self, state, action, s_prime):
        return self.transitions[state][action][s_prime]


    def step(self, action, state=None):
        self.env.set_state(state) if state else ...
        reward = self.env.step(action)
        is_done = self.env.is_done()
        return self.env.get_observation(), reward, is_done


    def random_strategy(self, _state):
        return random.choice(self.env.get_all_actions())

    def collect_episodes(self, n_samples, n_steps, exploration_strategy):
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

    def learn(self, n_samples: int, n_steps: int):
        episodes = self.collect_episodes(n_samples, n_steps, self.random_strategy)
        states = self.env.get_valid_states() + self.env._world.exit_pos
        actions = self.env.get_all_actions()

        transition_counts = {s: {a: { s_prime: 0 for s_prime in states} for a in actions} for s in states}
        reward_sums = {s: {a: { s_prime: 0 for s_prime in states} for a in actions} for s in states}
        
        for episode in episodes:
            for (s, a, r ,s_prime) in episode:
                if s_prime in states:
                    transition_counts[s][a][s_prime] += 1
                    reward_sums[s][a][s_prime] += r
                else:
                    print(r)
        
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

    def bellman_equation(self, state) -> float:
        actions = self.env.get_all_actions()
        values = np.zeros(len(actions))
        for action in actions:
            values[action] = sum(self.transitions[state][action][s_prime] * (self.rewards[state][action][s_prime] + (self.gamma * self.value_table[s_prime]))
                for s_prime in self.env.get_valid_states()+self.env._world.exit_pos
                )
        print(values)
        return max(values)
    

    def train(self,  n_updates: int):
        """
        Train the agent using value iteration for a specified number of updates

        Parameters:
        - n_updates (int): The total number of updates to perform
        """
        self.learn(100_000, 2_000)
        delta = 0
        for _ in range(n_updates):
            delta = self.value_iteration(delta)
            if delta <= self.theta:
                break

    def value_iteration(self, delta):
        for state in self.env.get_valid_states()+self.env._world.exit_pos:
            value = self.value_table[state]
            self.value_table[state] = self.bellman_equation(state)
            delta = max(delta, abs(value - self.value_table[state]))
        return delta


    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        # print(self.value_table)
        return self.value_table