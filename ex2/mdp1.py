from typing import Tuple

class Environment:
    def __init__(self):
        """
        Constructor of the Environment class.
        """
        self._initial_state = 1
        self._allowed_actions = [0, 1]  # 0: A, 1: B
        self._states = [1, 2, 3]
        self._current_state = self._initial_state

    def step(self, action: int) -> Tuple[int, int]:
        """
        Step function: compute the one-step dynamic from the given action.

        Args:
            action (int): the action taken by the agent.

        Returns:
            The tuple current_state, reward.
        """

        # check if the action is allowed
        if action not in self._allowed_actions:
            raise ValueError("Action is not allowed")

        reward = 0
        if action == 0 and self._current_state == 1:
            self._current_state = 2
            reward = 1
        elif action == 1 and self._current_state == 1:
            self._current_state = 3
            reward = 10
        elif action == 0 and self._current_state == 2:
            self._current_state = 1
            reward = 0
        elif action == 1 and self._current_state == 2:
            self._current_state = 3
            reward = 1
        elif action == 0 and self._current_state == 3:
            self._current_state = 2
            reward = 0
        elif action == 1 and self._current_state == 3:
            self._current_state = 3
            reward = 10

        return self._current_state, reward

    def reset(self) -> int:
        """
        Reset the environment starting from the initial state.

        Returns:
            The environment state after reset (initial state).
        """
        self._current_state = self._initial_state
        return self._current_state

env = Environment()
state = env.reset()

actions = [0, 0, 1, 1, 0, 1]

print(f"Initial state is {state}")

for action in actions:
    next_state, reward = env.step(action)
    print(f"From state {state} to state {next_state} with action {action}, reward: {reward}")
    state = next_state