import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.snake_env import SnakeEnv
from rl.model import DQN


# -----------------------------
# Helper: direction + relative actions
# -----------------------------
DIR_ORDER = ["UP", "RIGHT", "DOWN", "LEFT"]  # clockwise

def right_of(d): return DIR_ORDER[(DIR_ORDER.index(d) + 1) % 4]
def left_of(d):  return DIR_ORDER[(DIR_ORDER.index(d) - 1) % 4]

def apply_relative_action(current_dir: str, action: int) -> str:
    """
    action: 0=straight, 1=turn right, 2=turn left
    """
    if action == 0:
        return current_dir
    if action == 1:
        return right_of(current_dir)
    return left_of(current_dir)


# -----------------------------
# State features (11)
# -----------------------------
def _next_cell(head, direction):
    r, c = head
    if direction == "UP":    return (r - 1, c)
    if direction == "DOWN":  return (r + 1, c)
    if direction == "LEFT":  return (r, c - 1)
    return (r, c + 1)  # RIGHT

def _is_collision(state, cell):
    rows, cols = state["rows"], state["cols"]
    r, c = cell
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return True

    snake = state["snake"]
    head = snake[0]
    tail = snake[-1]

    # If the next cell is the tail, it's safe only if we are NOT eating food (tail moves away)
    will_grow = (cell == state["food"])
    body = set(snake)

    if cell in body:
        if cell == tail and not will_grow:
            return False
        if cell == head:
            return False
        return True

    return False

def get_observation(state) -> np.ndarray:
    """
    11 features:
    danger_straight, danger_right, danger_left (3)
    direction one-hot: up, right, down, left (4)
    food location relative to head: food_left, food_right, food_up, food_down (4)
    """
    snake = state["snake"]
    head = snake[0]
    dir_ = state["dir"]
    food = state["food"]

    straight = dir_
    right = right_of(dir_)
    left = left_of(dir_)

    danger_straight = 1.0 if _is_collision(state, _next_cell(head, straight)) else 0.0
    danger_right    = 1.0 if _is_collision(state, _next_cell(head, right)) else 0.0
    danger_left     = 1.0 if _is_collision(state, _next_cell(head, left)) else 0.0

    dir_up    = 1.0 if dir_ == "UP" else 0.0
    dir_right = 1.0 if dir_ == "RIGHT" else 0.0
    dir_down  = 1.0 if dir_ == "DOWN" else 0.0
    dir_left  = 1.0 if dir_ == "LEFT" else 0.0

    hr, hc = head
    fr, fc = food

    food_left  = 1.0 if fc < hc else 0.0
    food_right = 1.0 if fc > hc else 0.0
    food_up    = 1.0 if fr < hr else 0.0
    food_down  = 1.0 if fr > hr else 0.0

    return np.array([
        danger_straight, danger_right, danger_left,
        dir_up, dir_right, dir_down, dir_left,
        food_left, food_right, food_up, food_down
    ], dtype=np.float32)


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# -----------------------------
# Training
# -----------------------------
def train():
    # ----- Config -----
    rows, cols = 20, 20
    episodes = 1500
    max_steps_per_episode = 5000  # ends earlier due to death; this is just a cap

    gamma = 0.99
    lr = 1e-3
    batch_size = 128
    buffer_capacity = 80_000
    min_buffer = 2_000

    eps_start = 1.0
    eps_end = 0.05
    eps_decay_episodes = 900  # linearly decay over these many episodes

    target_update_every = 200  # gradient steps
    train_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Env / Models -----
    env = SnakeEnv(rows=rows, cols=cols)
    policy = DQN().to(device)
    target = DQN().to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    rb = ReplayBuffer(capacity=buffer_capacity)

    os.makedirs("rl/checkpoints", exist_ok=True)
    best_mean = -1e9
    recent_scores = deque(maxlen=50)

    grad_steps = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        obs = get_observation(state)

        # epsilon schedule (linear)
        t = min(1.0, ep / eps_decay_episodes)
        eps = eps_start + t * (eps_end - eps_start)

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps_per_episode):
            steps += 1

            # choose action
            if random.random() < eps:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q = policy(x)
                    action = int(torch.argmax(q, dim=1).item())

            # translate relative action -> absolute direction for env
            abs_dir = apply_relative_action(state["dir"], action)
            next_state, reward, done, info = env.step(abs_dir)
            next_obs = get_observation(next_state)

            rb.push(obs, action, reward, next_obs, done)

            obs = next_obs
            state = next_state
            total_reward += reward

            # learn
            if len(rb) >= min_buffer and (steps % train_every == 0):
                s, a, r, s2, d = rb.sample(batch_size)

                s_t  = torch.tensor(s, dtype=torch.float32, device=device)
                a_t  = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r_t  = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                s2_t = torch.tensor(s2, dtype=torch.float32, device=device)
                d_t  = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                q_sa = policy(s_t).gather(1, a_t)

                with torch.no_grad():
                    # standard DQN target
                    max_q_next = target(s2_t).max(dim=1, keepdim=True)[0]
                    y = r_t + gamma * (1.0 - d_t) * max_q_next

                loss = loss_fn(q_sa, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                optimizer.step()

                grad_steps += 1
                if grad_steps % target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

            if done:
                break

        score = env.score
        recent_scores.append(score)
        mean50 = float(np.mean(recent_scores))

        if ep % 25 == 0:
            print(f"ep {ep:4d} | score={score:3d} | mean50={mean50:.2f} | eps={eps:.2f} | steps={steps}")

        # save best
        if len(recent_scores) == recent_scores.maxlen and mean50 > best_mean:
            best_mean = mean50
            torch.save(policy.state_dict(), "rl/checkpoints/snake_dqn.pt")
            print(f"saved new best model | mean50={best_mean:.2f}")

    # final save (in case best never triggered)
    if not os.path.exists("rl/checkpoints/snake_dqn.pt"):
        torch.save(policy.state_dict(), "rl/checkpoints/snake_dqn.pt")
        print("saved model to rl/checkpoints/snake_dqn.pt")

    print("Training done.")


if __name__ == "__main__":
    train()