import random
from collections import deque

DIRS = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
}

OPPOSITE = {
    "UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"
}

class SnakeEnv:
    def __init__(self, rows=20, cols=20, seed=None):
        self.rows = rows
        self.cols = cols
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        r = self.rows // 2
        c = self.cols // 2
        self.snake = deque([(r, c), (r, c-1), (r, c-2)])  # head at [0]
        self.dir = "RIGHT"
        self.score = 0
        self.done = False
        self._place_food()
        return self.get_state()

    def _place_food(self):
        empty = [(r, c) for r in range(self.rows) for c in range(self.cols)
                 if (r, c) not in self.snake]
        if not empty:
            # snake filled the board
            self.done = True
            self.food = None
            return
        self.food = self.rng.choice(empty)

    def step(self, action_dir):
        """
        action_dir: "UP"/"DOWN"/"LEFT"/"RIGHT"
        returns: (state, reward, done, info)
        """
        if self.done:
            return self.get_state(), 0.0, True, {}

        # prevent instant reverse
        if action_dir in DIRS and action_dir != OPPOSITE[self.dir]:
            self.dir = action_dir

        dr, dc = DIRS[self.dir]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        # collision with wall
        if not (0 <= new_head[0] < self.rows and 0 <= new_head[1] < self.cols):
            self.done = True
            return self.get_state(), -10.0, True, {"event": "wall"}

        # collision with self (tail is allowed only if we move tail away)
        tail = self.snake[-1]
        body_set = set(self.snake)
        will_grow = (new_head == self.food)
        if new_head in body_set and not (new_head == tail and not will_grow):
            self.done = True
            return self.get_state(), -10.0, True, {"event": "self"}

        # move
        self.snake.appendleft(new_head)

        reward = 0.0
        if new_head == self.food:
            self.score += 1
            reward = +10.0
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.01  # tiny step penalty to encourage efficiency

        # if board filled, mark done
        if self.food is None:
            return self.get_state(), +50.0, True, {"event": "filled_board"}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Convenient for pygame + later RL. Returns dict."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "snake": list(self.snake),
            "food": self.food,
            "dir": self.dir,
            "score": self.score,
            "done": self.done,
        }