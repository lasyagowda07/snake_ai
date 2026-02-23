import time
import numpy as np
import torch

from core.snake_env import SnakeEnv
from rl.model import DQN
from rl.train_dqn import get_observation, apply_relative_action


def run_games(num_games=10, delay=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeEnv(rows=20, cols=20)
    model = DQN().to(device)
    model.load_state_dict(torch.load("rl/checkpoints/snake_dqn.pt", map_location=device))
    model.eval()

    scores = []

    for g in range(1, num_games + 1):
        state = env.reset()
        obs = get_observation(state)

        while True:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(model(x), dim=1).item())

            abs_dir = apply_relative_action(state["dir"], action)
            state, reward, done, info = env.step(abs_dir)
            obs = get_observation(state)

            if delay > 0:
                time.sleep(delay)

            if done:
                scores.append(env.score)
                print(f"game {g}/{num_games} score = {env.score}")
                break

    print(f"avg score: {np.mean(scores):.2f} | max: {np.max(scores)}")


if __name__ == "__main__":
    run_games(num_games=20, delay=0.0)