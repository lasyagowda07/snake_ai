# web3d_backend/server_ws.py
import asyncio
import json
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from core.snake3d_env import Snake3DEnv
from rl3d.model3d import DQN3D
from rl3d.train_dqn3d import get_observation

app = FastAPI()

FPS = 10
SIZE = (10, 10, 10)
MODEL_PATH = "rl3d/checkpoints/snake3d_dqn.pt"


class GameStreamer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = Snake3DEnv(*SIZE)
        self.model = DQN3D().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

        self.state = self.env.reset()
        self.obs = get_observation(self.state)

    def step(self):
        with torch.no_grad():
            x = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = int(torch.argmax(self.model(x), dim=1).item())

        self.state, reward, done, info = self.env.step(action)
        self.obs = get_observation(self.state)

        if done:
            self.state = self.env.reset()
            self.obs = get_observation(self.state)

        return self.state


streamer = GameStreamer()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            state = streamer.step()
            payload = {
                "size": [state["size_x"], state["size_y"], state["size_z"]],
                "snake": state["snake"],  # list of [x,y,z]
                "food": state["food"],    # [x,y,z]
                "dir": state["dir"],
                "score": state["score"],
            }
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(1.0 / FPS)
    except WebSocketDisconnect:
        return