import lle
import numpy as np
from .env import Labyrinth
from collections import defaultdict
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
        self.theta = 0.001
        self.env_size = env.get_map_size()
        self.value_table = np.zeros(self.env_size)
        self.rewards = {}
        self.step_count = 0
        self.transitions = {}


    def transition(self, state, action, update=False):
        reward = self.env.step(action)
        s_prime = self.env.get_observation()
        
        # Check if we reached the exit (terminal state)
        if self.env.is_done():
            reward = 100  # Large reward for reaching the exit
        
        self.env.step(lle.Action(action).opposite().value)  # Reset to original state
        
        if update:
            # Learning the Transitions
            if (state, action) not in self.transitions:
                self.transitions[(state, action)] = []
            
            # Update or add new transition
            transition_found = False
            for transition in self.transitions[(state, action)]:
                if transition[0] == s_prime:
                    transition[1] += 1
                    transition_found = True
                    break
            
            if not transition_found:
                self.transitions[(state, action)].append([s_prime, 1, 0])
            
            # Update probabilities for all transitions from this state-action pair
            total_visits = sum(t[1] for t in self.transitions[(state, action)])
            for transition in self.transitions[(state, action)]:
                transition[2] = transition[1] / total_visits
            
            # Learning the Rewards
            if (state, action, s_prime) not in self.rewards:
                self.rewards[(state, action, s_prime)] = [0, 0, 0]
            
            self.rewards[(state, action, s_prime)][0] += reward
            self.rewards[(state, action, s_prime)][1] += 1
            self.rewards[(state, action, s_prime)][2] = (
                self.rewards[(state, action, s_prime)][0] / 
                self.rewards[(state, action, s_prime)][1]
            )
        
        # Reading Values - Handle case when transition hasn't been seen
        probability = 0
        if (state, action) in self.transitions:
            for transition in self.transitions[(state, action)]:
                if transition[0] == s_prime:
                    probability = transition[2]
                    break
        
        reward_value = 0
        if (state, action, s_prime) in self.rewards:
            reward_value = self.rewards[(state, action, s_prime)][2]
        else:
            reward_value = reward  # Use immediate reward if no history
        
        return s_prime, probability, reward_value


    def bellman_equation(self, state) -> float:
        values = np.zeros(len(self.env.get_all_actions()))
        for action in self.env.get_all_actions():
            s_prime, probability, reward = self.transition(state, action, True)
            values[action] += probability * (reward + (self.gamma ** self.step_count) * self.value_table[s_prime])
        return max(values)
            

    def train(self,  n_updates: int):
        """
        Train the agent using value iteration for a specified number of updates.

        Parameters:
        - n_updates (int): The total number of updates to perform.
        """
        for _ in range(n_updates):
            delta = 0
            for state in self.env.get_valid_states():
                value = self.value_table[state]
                self.value_table[state] = self.bellman_equation(state)
                delta = max(delta, abs(value - self.value_table[state]))
                self.step_count += 1
                if delta < self.theta:
                    break



    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        # print(self.value_table)
        return self.value_table