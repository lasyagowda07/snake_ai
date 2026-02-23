# rl3d/train_dqn3d.py
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.snake3d_env import Snake3DEnv, DIRS
from rl3d.model3d import DQN3D


# -----------------------------
# Observation (18 dims)
# -----------------------------
def _next_cell(head, dir_id):
    dx, dy, dz = DIRS[dir_id]
    x, y, z = head
    return (x + dx, y + dy, z + dz)

def _is_collision(state, cell):
    sx, sy, sz = state["size_x"], state["size_y"], state["size_z"]
    x, y, z = cell
    if x < 0 or x >= sx or y < 0 or y >= sy or z < 0 or z >= sz:
        return True

    snake = state["snake"]
    head = snake[0]
    tail = snake[-1]
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
    snake = state["snake"]
    head = snake[0]
    dir_id = state["dir"]
    food = state["food"]

    # danger in 6 absolute directions
    danger = []
    for d in range(6):
        danger.append(1.0 if _is_collision(state, _next_cell(head, d)) else 0.0)

    # direction one-hot (6)
    dir_oh = [1.0 if dir_id == d else 0.0 for d in range(6)]

    # food relative signs (6)
    hx, hy, hz = head
    fx, fy, fz = food
    food_rel = [
        1.0 if fx > hx else 0.0,  # +x
        1.0 if fx < hx else 0.0,  # -x
        1.0 if fy > hy else 0.0,  # +y
        1.0 if fy < hy else 0.0,  # -y
        1.0 if fz > hz else 0.0,  # +z
        1.0 if fz < hz else 0.0,  # -z
    ]

    return np.array(danger + dir_oh + food_rel, dtype=np.float32)


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=120_000):
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


def train():
    # -------- Config --------
    size_x, size_y, size_z = 10, 10, 10  # start small; scale later
    episodes = 3000
    max_steps_per_episode = 3000

    gamma = 0.99
    lr = 1e-3
    batch_size = 256
    buffer_capacity = 200_000
    min_buffer = 5_000

    eps_start = 1.0
    eps_end = 0.05
    eps_decay_episodes = 1800

    target_update_every = 400  # gradient steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Snake3DEnv(size_x=size_x, size_y=size_y, size_z=size_z)

    policy = DQN3D(input_dim=18, hidden=256, output_dim=6).to(device)
    target = DQN3D(input_dim=18, hidden=256, output_dim=6).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    opt = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    rb = ReplayBuffer(buffer_capacity)

    os.makedirs("rl3d/checkpoints", exist_ok=True)

    recent = deque(maxlen=50)
    best_mean = -1e9
    grad_steps = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        obs = get_observation(state)
        total_reward = 0.0

        t = min(1.0, ep / eps_decay_episodes)
        eps = eps_start + t * (eps_end - eps_start)

        for step in range(max_steps_per_episode):
            # epsilon-greedy
            if random.random() < eps:
                action = random.randint(0, 5)
            else:
                with torch.no_grad():
                    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q = policy(x)
                    action = int(torch.argmax(q, dim=1).item())

            next_state, reward, done, info = env.step(action)
            next_obs = get_observation(next_state)

            rb.push(obs, action, reward, next_obs, done)

            obs = next_obs
            state = next_state
            total_reward += reward

            # learn
            if len(rb) >= min_buffer:
                s, a, r, s2, d = rb.sample(batch_size)

                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
                s2_t = torch.tensor(s2, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)

                q_sa = policy(s_t).gather(1, a_t)
                with torch.no_grad():
                    max_q_next = target(s2_t).max(dim=1, keepdim=True)[0]
                    y = r_t + gamma * (1.0 - d_t) * max_q_next

                loss = loss_fn(q_sa, y)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                opt.step()

                grad_steps += 1
                if grad_steps % target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

            if done:
                break

        score = env.score
        recent.append(score)
        mean50 = float(np.mean(recent))

        if ep % 50 == 0:
            print(f"ep {ep:4d} | score={score:3d} | mean50={mean50:.2f} | eps={eps:.2f}")

        if len(recent) == recent.maxlen and mean50 > best_mean:
            best_mean = mean50
            torch.save(policy.state_dict(), "rl3d/checkpoints/snake3d_dqn.pt")
            print(f"saved best | mean50={best_mean:.2f}")

    # final save if none
    if not os.path.exists("rl3d/checkpoints/snake3d_dqn.pt"):
        torch.save(policy.state_dict(), "rl3d/checkpoints/snake3d_dqn.pt")

    print("Training done.")


if __name__ == "__main__":
    train()