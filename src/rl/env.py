from itertools import product
import random
import numpy as np
import lle
BOTTOM_LEFT_EXIT = (6,0)

class Labyrinth:
    ACTION_SYMBOLS = ['↑', '↓', '→', '←']  # NORTH, SOUTH, EAST, WEST
    def __init__(self, malfunction_probability: float = 0.0):
        """
        Initialize the Labyrinth environment.

        Parameters:
        - malfunction_probability (float): Probability of the agent malfunctioning, 
                                           which causes it to teleport to a random valid position.
        """
        self.malfunction_probability = malfunction_probability
        self._world = lle.World("""
                .  .  .  X  @  .  S0
                .  @  .  .  .  .  @
                .  @  .  .  .  @  .
                .  .  @  .  @  .  .
                @  .  @  .  .  @  .
                .  .  .  @  .  .  .
                X  @  .  .  @  .  .""")
        
        self._done = False
        self._first_render = True
        all_positions = set(product(range(self._world.height), range(self._world.width)))
        self._valid_positions = list(all_positions - set(self._world.wall_pos) - set(self._world.exit_pos))

    def get_map_size(self) -> tuple[int, int]:
        """
        Retrieve the size of the labyrinth map.

        Returns:
        - tuple[int, int]: A tuple containing the height and width of the labyrinth (height, width).
        """
        return (self._world.height, self._world.width)
        
    def set_state(self, state: tuple[int, int]) -> None:
        """
        Set the agent's state (position) in the world.

        Parameters:
        - state (tuple[int, int]): The (row, column) position to place the agent in.
        """
        self._world.set_state(lle.WorldState([state], []))

    def get_valid_states(self) -> list[tuple[int, int]]:
        """
        Retrieve a list of valid positions for the agent.

        Returns:
        - list[tuple[int, int]]: List of valid positions (not walls or exits).
        """
        return self._valid_positions

    def reset(self) -> tuple[int, int]:
        """
        Reset the labyrinth to its initial state.

        Returns:
        - tuple[int, int]: The initial observation (agent's starting position).
        """
        self._done = False
        self._world.reset()
        return self.get_observation()
        
    def get_all_actions(self) -> list[int]:
        """
        Retrieve all possible actions.

        Returns:
        - list[int]: List of all possible action indices.
        """
        return list(range(len(Labyrinth.ACTION_SYMBOLS)))

    def step(self, action: int, state: tuple[int, int] = None) -> float:
        """
        Perform an action in the environment and return the associated reward.

        Parameters:
        - action (int): The action index to be performed.
        - state (tuple[int, int], optional): Optional state to set the agent to before acting.

        Returns:
        - float: Reward received after performing the action.
        """
        if state is not None:
            self.set_state(state)
        
        if np.random.uniform() < self.malfunction_probability:
            random_state = random.choice(self.get_valid_states())
            self.set_state(random_state)
            action = lle.Action.STAY
        else:
            action = self._validate_action(action)

        reward = self._execute_action(action)
        return reward

    def _validate_action(self, action: int) -> lle.Action:
        """
        Validate and adjust the action if it's unavailable.

        Parameters:
        - action (int): The action index to validate.

        Returns:
        - lle.Action: The valid action or lle.Action.STAY if unavailable.
        """
        if lle.Action(action) in self._world.available_actions()[0]:
            return lle.Action(action)
        return lle.Action.STAY

    def _execute_action(self, action: lle.Action) -> float:
        """
        Execute the specified action and determine reward based on events.

        Parameters:
        - action (lle.Action): The validated action to execute.

        Returns:
        - float: Reward based on action result.
        """
        events = self._world.step([action])
        for event in events:
            if event.event_type == lle.EventType.AGENT_EXIT:
                self._done = True
                return 100 if self.get_observation() == BOTTOM_LEFT_EXIT else 10
        return -1

    def get_observation(self) -> tuple[int, int]:
        """
        Get the agent's current position in the labyrinth.

        Returns:
        - tuple[int, int]: The (row, column) position of the agent.
        """
        return self._world.agents_positions[0]

    def is_done(self) -> bool:
        """
        Check if a terminal state has been reached.

        Returns:
        - bool: True if a terminal state has been reached, False otherwise.
        """
        return self._done

    def render(self) -> None:
        """
        Render the labyrinth environment using OpenCV.
        """
        import cv2

        img = self._world.get_image()
        if self._first_render:
            # Solves a bug such that the first rendering is not displayed correctly the first time
            cv2.imshow("Labyrinth", img)
            cv2.waitKey(1)
            self._first_render = False
            import time

            time.sleep(0.2)

        cv2.imshow("Labyrinth", img)
        cv2.waitKey(1)
