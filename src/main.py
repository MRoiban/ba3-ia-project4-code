from rl.env import Labyrinth
from rl.qlearning import QLearning
from rl.value_iteration import ValueIteration
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List
# from rl.qlearning import train_multiple_agents


def plot_values(values: NDArray[np.float64]) -> None:
    """
    Plots a heatmap representing the state values in a grid world.

    Parameters:
    - values (NDArray[np.float64]): A 2D numpy array of shape (height, width) where each element 
                                    represents the computed value of that state.

    Returns:
    - None: Displays the plot.
    """
    assert values.ndim == 2, f"Expected 2D array of shape (height, width), got shape {values.shape}"
    height, width = values.shape
    
    # Create the base heatmap
    # plt.figure(figsize=(8, 8))
    plt.imshow(values, origin="upper")
    plt.colorbar(label="State Value")
    
    # Overlay the actual values
    for i in range(height):
        for j in range(width):
            plt.text(j, i, f'{values[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='black', fontsize=10)
    
    plt.title('State Values Heatmap')
    plt.grid(False)
    plt.show()

def plot_qvalues(q_values: NDArray[np.float64], action_symbols: list[str]) -> None:
    """
    Plots a heatmap of the maximum Q-values in each state of a grid world and overlays symbols
    to represent the optimal action in each state.

    Parameters:
    - q_values (NDArray[np.float64]): A 3D numpy array of shape (height, width, n_actions), where each cell contains Q-values
                                      for four possible actions (up, down, right, left).
    - env (Labyrinth): The environment instance to access action symbols.

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

def plot_training_results(all_episode_rewards: List[List[float]], title: str = "Training Results") -> None:
    """
    Plot the average rewards and confidence interval across multiple training runs.
    
    Parameters:
    - all_episode_rewards: List of reward lists from multiple training runs
    - title: Title for the plot
    """
    # Convert episodes of different lengths to fixed-size array
    max_episodes = max(len(rewards) for rewards in all_episode_rewards)
    rewards_array = np.zeros((len(all_episode_rewards), max_episodes))
    
    # Fill the array, padding shorter episodes with NaN
    for i, rewards in enumerate(all_episode_rewards):
        rewards_array[i, :len(rewards)] = rewards
        rewards_array[i, len(rewards):] = np.nan
    
    # Calculate mean and standard deviation, ignoring NaN values
    mean_rewards = np.nanmean(rewards_array, axis=0)
    std_rewards = np.nanstd(rewards_array, axis=0)
    
    # Create x-axis (episode numbers)
    episodes = np.arange(1, max_episodes + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards, label='Mean Reward', color='blue')
    
    # Add confidence interval
    plt.fill_between(episodes, 
                    mean_rewards - std_rewards,
                    mean_rewards + std_rewards,
                    alpha=0.2,
                    color='blue')
    
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    
    env = Labyrinth(malfunction_probability = 0.1)

    env.reset()
    env.render()

    # Uncomment for random moves
    random_moves(env,10)
    
    # Uncomment for Value Iteration
    # algo = ValueIteration(env=env, gamma=.9)
    # algo.train(50)
    # plot_values(algo.get_value_table())

    # Uncomment for Q-learning
    # algo = QLearning(env=env,alpha=.1,epsilon=.1, gamma=0.9)
    algo = QLearning(env=env,alpha=.1,c=10, gamma=0.9)
    algo.train(10_000)
    plot_qvalues(algo.get_q_table(),action_symbols=Labyrinth.ACTION_SYMBOLS)

    # Uncomment for training multiple agents and plotting results
    # results = train_multiple_agents(
    #     env,
    #     n_agents=20,
    #     n_steps=10000,
    #     gamma=0.9,
    #     alpha=0.1,
    #     epsilon=0.1
    # )
    # plot_training_results(results, "Q-Learning Training Results (ε-greedy)")
