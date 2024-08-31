class Environment:
    def __init__(self):
        self._initial_state = 1
        self._allowed_actions = [0, 1]
        self._states = [1, 2, 3]
        self._current_state = self._initial_state

    def step(self, action):
        if action not in self._allowed_actions:
            raise ValueError("Action is not allowed")

        reward = 0

        # Define state transitions and rewards
        transitions = {
            (1, 0): (2, 1),
            (1, 1): (3, 10),
            (2, 0): (2, 0),
            (2, 1): (3, 1),
            (3, 0): (1, 0),
            (3, 1): (3, 10)
        }

        if (self._current_state, action) in transitions:
            self._current_state, reward = transitions[(self._current_state, action)]

        return self._current_state, reward

    def reset(self):
        self._current_state = self._initial_state
        return self._current_state

env = Environment()
state = env.reset()
actions = [0,0,1,1,0,1]
print(f"Initial State is {state}")
for action in actions:
    next_state, reward = env.step(action=action)
    print(f"From state {state} to {next_state} with action {action}, reward {reward}")
    state = next_state