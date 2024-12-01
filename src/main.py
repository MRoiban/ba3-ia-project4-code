from rl.env import Labyrinth
from rl.qlearning import QLearning
from rl.value_iteration import ValueIteration
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray
from typing import List
import argparse


def plot_values(values: NDArray[np.float64], env: Labyrinth = None) -> None:
    """
    Plots a heatmap representing the state values in a grid world.

    Parameters:
    - values (NDArray[np.float64]): A 2D numpy array of shape (height, width) where each element 
                                    represents the computed value of that state.
    - env (Labyrinth): The environment instance to access exit positions.

    Returns:
    - None: Displays the plot.
    """
    assert values.ndim == 2, f"Expected 2D array of shape (height, width), got shape {values.shape}"
    sns.heatmap(values, annot=True, cbar_kws={'label': 'Value'})
    
    # Add flag symbols at exit positions
    if env is not None:
        for x, y in env._world.exit_pos:
            plt.text(y + 0.5, x + 0.5, '⚑', ha='center', va='center', color='red', fontsize=15)
    plt.title("Value Heatmap")
    plt.grid(False)
    plt.show()

def plot_qvalues(q_values: NDArray[np.float64], action_symbols: list[str], env: Labyrinth = None) -> None:
    """
    Plots a heatmap of the maximum Q-values in each state of a grid world and overlays symbols
    to represent the optimal action in each state.

    Parameters:
    - q_values (NDArray[np.float64]): A 3D numpy array of shape (height, width, n_actions), where each cell contains Q-values
                                      for four possible actions (up, down, right, left).
    - action_symbols (list[str]): List of symbols representing each action.
    - env (Labyrinth): The environment instance to access exit positions.

    Returns:
    - None: Displays the plot.
    """
    assert q_values.ndim == 3, f"Expected 3D array of shape (height, width, n_actions), got shape {q_values.shape}"
    assert q_values.shape[-1] == len(action_symbols), f"Number of action symbols should match the number of actions"
    height, width = q_values.shape[:2]

    # Calculate the best action and max Q-value for each cell
    best_actions = np.argmax(q_values, axis=2)
    max_q_values = np.max(q_values, axis=2)

    # Plotting the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(max_q_values, origin="upper")
    plt.colorbar(label="Max Q-value")
    # Overlay best action symbols
    for i in range(height):
        for j in range(width):
            action_symbol = action_symbols[best_actions[i, j]]
            plt.text(j, i, action_symbol, ha='center', va='center', color='black', fontsize=12)

    # Add flag symbols at exit positions
    if env is not None:
        for x, y in env._world.exit_pos:
            plt.text(y, x, '⚑', ha='center', va='center', color='red', fontsize=15)

    # Labels and layout
    plt.title("Q-value Heatmap with Optimal Actions")
    plt.grid(False)
    plt.show()


def random_moves(env: Labyrinth, n_steps: int) -> None:
    """
    Makes random moves in the environment and renders each step.

    Parameters:
    - env (Labyrinth): The environment instance where random moves will be performed.
    - n_steps (int): Number of random steps to perform.

    Returns:
    - None
    """
    env.reset()
    env.render()
    episode_rewards = 0
    for s in range(n_steps):
        random_action = np.random.choice(env.get_all_actions())
        reward = env.step(random_action)
        done = env.is_done()
        episode_rewards += reward
        if done:
            print("collected reward =", episode_rewards)
            env.reset()
            episode_rewards = 0
        env.render()


def plot_training_results(all_episode_rewards: List[List[float]], labels: List[str], title: str = "Training Results") -> None:
    """
    Plot the average rewards and confidence interval across multiple training runs.
    
    Parameters:
    - all_episode_rewards: List of lists of reward lists from multiple training runs
    - labels: Labels for each configuration
    - title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    for i, rewards_list in enumerate(all_episode_rewards):
        # Convert episodes of different lengths to fixed-size array
        max_episodes = max(len(rewards) for rewards in rewards_list)
        rewards_array = np.zeros((len(rewards_list), max_episodes))
        
        # Fill the array, padding shorter episodes with NaN
        for j, rewards in enumerate(rewards_list):
            rewards_array[j, :len(rewards)] = rewards
            rewards_array[j, len(rewards):] = np.nan
        
        # Calculate mean and standard deviation, ignoring NaN values
        mean_rewards = np.nanmean(rewards_array, axis=0)
        std_rewards = np.nanstd(rewards_array, axis=0)
        
        # Create x-axis (episode numbers)
        episodes = np.arange(1, max_episodes + 1)
        
        plt.plot(episodes, mean_rewards, label=labels[i])
        plt.fill_between(episodes, 
                        mean_rewards - std_rewards,
                        mean_rewards + std_rewards,
                        alpha=0.2)
    
    plt.xlabel('Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def value_iteration_tests(plot_values, env):
    value_iteration_step_tests(plot_values, env, 10)
    value_iteration_step_tests(plot_values, env, 20)
    value_iteration_step_tests(plot_values, env, 30)
    value_iteration_step_tests(plot_values, env, 40)
    value_iteration_step_tests(plot_values, env, 50)

def value_iteration_step_tests(plot_values, env, steps):
    algo = ValueIteration(env=env, gamma=1)
    algo.train(steps)
    plot_values(algo.get_value_table(), env)

def qlearning_tests(plot_qvalues, env):
    algo = QLearning(env=env, alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma)
    algo.train(args.steps)
    plot_qvalues(algo.get_q_table(), action_symbols=Labyrinth.ACTION_SYMBOLS, env=env)

if __name__ == "__main__":
  
    env = Labyrinth(malfunction_probability=0.1)
    env.reset()

    

    value_iteration_tests(plot_values, env)
    
    # qlearning_tests(plot_qvalues, env)
