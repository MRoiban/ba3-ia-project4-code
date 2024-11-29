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
        self.theta = 0.001
        self.env_size = env.get_map_size()
        self.value_table = np.zeros(self.env_size)
        self.rewards = {}
        self.step_count = 0
        self.state_visits = np.zeros(self.env_size)
        self.transitions = {}


    # def transition(self, state, action, update=False):
    #     reward = self.env.step(action)
    #     s_prime = self.env.get_observation()
        
    #     self.env.step(lle.Action(action).opposite().value)  # Reset to original state
        
    #     if update:
    #         # Learning the Transitions
    #         if (state, action) not in self.transitions:
    #             self.transitions[(state, action)] = []
            
    #         # Update or add new transition
    #         transition_found = False
    #         for transition in self.transitions[(state, action)]:
    #             if transition[0] == s_prime:
    #                 transition[1] += 1
    #                 transition_found = True
    #                 break
            
    #         if not transition_found:
    #             self.transitions[(state, action)].append([s_prime, 1, 0])
            
    #         # Update probabilities for all transitions from this state-action pair
    #         total_visits = 0
    #         for _, visit_count, _ in self.transitions[(state, action)]:
    #             total_visits += visit_count
    #         for transition in self.transitions[(state, action)]:
    #             transition[2] = transition[1] / total_visits  # Update probability
            
            
    #         # Learning the Rewards
    #         if (state, action, s_prime) not in self.rewards:
    #             self.rewards[(state, action, s_prime)] = [0, 0, 0]
            
    #         self.rewards[(state, action, s_prime)][0] += reward
    #         self.rewards[(state, action, s_prime)][1] += 1
    #         self.rewards[(state, action, s_prime)][2] = (
    #             self.rewards[(state, action, s_prime)][0] / 
    #             self.rewards[(state, action, s_prime)][1]
    #         )
        
    #     # Reading Values - Handle case when transition hasn't been seen
    #     probability = 0
    #     if (state, action) in self.transitions:
    #         for transition in self.transitions[(state, action)]:
    #             if transition[0] == s_prime:
    #                 probability = transition[2]
    #                 break
        
    #     reward_value = 0
    #     if (state, action, s_prime) in self.rewards:
    #         reward_value = self.rewards[(state, action, s_prime)][2]
    #     else:
    #         reward_value = reward  # Use immediate reward if no history

    def transition(self, state, action, update=False):
        reward = self.env.step(action)
        s_prime = self.env.get_observation()
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


    def learn_transitions(self, n_samples: int):
        """
        Learn the transition function by interacting with the environment.

        Parameters:
        - n_samples (int): The number of state-action pairs to sample and update.
        """
        for _ in range(n_samples):
            # Randomly sample a valid state and action
            state = self.env.get_valid_states()[random.randint(0, len(self.env.get_valid_states()) - 1)]
            self.env.set_state(state)
            self.state_visits[state] += 1

            action = self.env.get_all_actions()[random.randint(0, len(self.env.get_all_actions()) - 1)]

            # Take the action and observe the next state and reward
            reward = self.env.step(action)
            s_prime = self.env.get_observation()


            # Reset the environment to the original state
            self.env.reset()

            # Update transition counts and probabilities
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
                transition[2] = transition[1] / total_visits  # Update probability

            # Update the reward table
            if (state, action, s_prime) not in self.rewards:
                self.rewards[(state, action, s_prime)] = [0, 0, 0]

            self.rewards[(state, action, s_prime)][0] += reward
            self.rewards[(state, action, s_prime)][1] += 1
            self.rewards[(state, action, s_prime)][2] = (
                self.rewards[(state, action, s_prime)][0] / 
                self.rewards[(state, action, s_prime)][1]
            )

    def bellman_equation(self, state) -> float:
        values = np.zeros(len(self.env.get_all_actions()))
        for action in self.env.get_all_actions():
            s_prime, probability, reward = self.transition(state, action, True)
            values[action] += probability * (reward + (self.gamma) * self.value_table[s_prime])
        return max(values)
    
    def visualize(self, samples: int):
        import matplotlib.pyplot as plt
        import numpy as np

        # Initialize a 7x7 grid
        grid_size = 7

        # Plotting the state visit grid and heatmap
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # State visit grid plot
        ax[0].imshow(self.state_visits, cmap="Greys", origin="lower")
        ax[0].set_title("State Visit Grid (7x7)")
        ax[0].set_xticks(range(grid_size))
        ax[0].set_yticks(range(grid_size))
        ax[0].grid(visible=True, color='black', linewidth=0.5)
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Y")

        # Heatmap of state visits
        heatmap = ax[1].imshow(self.state_visits, cmap="hot", origin="lower")
        ax[1].set_title("Heatmap of State Visits")
        ax[1].set_xticks(range(grid_size))
        ax[1].set_yticks(range(grid_size))
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Y")

        # Add a colorbar to the heatmap
        plt.colorbar(heatmap, ax=ax[1], orientation='vertical')

        plt.tight_layout()
        plt.show()

    def train(self,  n_updates: int):
        """
        Train the agent using value iteration for a specified number of updates

        Parameters:
        - n_updates (int): The total number of updates to perform
        """
        self.learn_transitions(1_000_000)
        self.visualize(1_000_000)
        for _ in range(n_updates):
            delta = 0
            updated_values = np.zeros(self.env_size)
            for state in self.env.get_valid_states():
                value = self.value_table[state]
                updated_value = self.bellman_equation(state)
                updated_values[state] = updated_value
                delta = max(delta, abs(value - updated_value))
                self.step_count += 1
                if delta < self.theta:
                    break
            self.value_table = updated_values



    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        # print(self.value_table)
        return self.value_table