# core/snake3d_env.py
import random
from collections import deque

DIRS = {
    0: ( 1,  0,  0),  # +X
    1: (-1,  0,  0),  # -X
    2: ( 0,  1,  0),  # +Y
    3: ( 0, -1,  0),  # -Y
    4: ( 0,  0,  1),  # +Z
    5: ( 0,  0, -1),  # -Z
}

OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}


class Snake3DEnv:
    """
    3D Snake:
    - Coordinates are (x, y, z)
    - Snake is deque: head at index 0
    """
    def __init__(self, size_x=10, size_y=10, size_z=10, seed=None):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        cx = self.size_x // 2
        cy = self.size_y // 2
        cz = self.size_z // 2

        # Start length 3 along -X
        self.snake = deque([(cx, cy, cz), (cx-1, cy, cz), (cx-2, cy, cz)])
        self.dir = 0  # +X
        self.score = 0
        self.done = False
        self.steps = 0

        self._place_food()
        return self.get_state()

    def _place_food(self):
        snake_set = set(self.snake)
        empty = []
        for x in range(self.size_x):
            for y in range(self.size_y):
                for z in range(self.size_z):
                    if (x, y, z) not in snake_set:
                        empty.append((x, y, z))

        if not empty:
            self.done = True
            self.food = None
            return

        self.food = self.rng.choice(empty)

    def _in_bounds(self, p):
        x, y, z = p
        return (0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z)

    def step(self, action_dir: int):
        """
        action_dir: 0..5 (absolute direction)
        returns: (state, reward, done, info)
        """
        if self.done:
            return self.get_state(), 0.0, True, {}

        self.steps += 1

        # prevent instant reverse
        if action_dir in DIRS and action_dir != OPPOSITE[self.dir]:
            self.dir = action_dir

        dx, dy, dz = DIRS[self.dir]
        hx, hy, hz = self.snake[0]
        new_head = (hx + dx, hy + dy, hz + dz)

        # wall
        if not self._in_bounds(new_head):
            self.done = True
            return self.get_state(), -10.0, True, {"event": "wall"}

        snake_set = set(self.snake)
        tail = self.snake[-1]
        will_grow = (new_head == self.food)

        # self-collision (tail is safe if not growing)
        if new_head in snake_set and not (new_head == tail and not will_grow):
            self.done = True
            return self.get_state(), -10.0, True, {"event": "self"}

        # move
        self.snake.appendleft(new_head)

        if will_grow:
            self.score += 1
            reward = +10.0
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.01

        # filled board
        if self.food is None:
            return self.get_state(), +50.0, True, {"event": "filled_board"}

        return self.get_state(), reward, False, {}

    def get_state(self):
        return {
            "size_x": self.size_x,
            "size_y": self.size_y,
            "size_z": self.size_z,
            "snake": list(self.snake),
            "food": self.food,
            "dir": self.dir,
            "score": self.score,
            "done": self.done,
            "steps": self.steps,
        }