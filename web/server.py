import io
import time
import threading

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from core.snake_env import SnakeEnv
from rl.model import DQN
from rl.train_dqn import get_observation, apply_relative_action

from PIL import Image, ImageDraw, ImageFont


# -----------------------
# Config
# -----------------------
ROWS, COLS = 20, 20
CELL = 20  # pixel size per grid cell
FPS = 20
MODEL_PATH = "rl/checkpoints/snake_dqn.pt"

# colors
BG = (18, 18, 18)
GRID = (30, 30, 30)
SNAKE_HEAD = (80, 220, 120)
SNAKE_BODY = (60, 170, 100)
FOOD = (255, 90, 90)
TEXT = (240, 240, 240)

app = FastAPI()

latest_jpeg = None
latest_lock = threading.Lock()


def render_frame(state) -> bytes:
    """Render env state to JPEG bytes."""
    w = COLS * CELL
    h = ROWS * CELL + 30  # extra for score bar
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    # score bar
    score_text = f"score: {state['score']}"
    draw.text((8, 5), score_text, fill=TEXT)

    # grid + entities (offset y by 30)
    y0 = 30

    # grid lines (light)
    for r in range(ROWS + 1):
        draw.line([(0, y0 + r * CELL), (w, y0 + r * CELL)], fill=GRID)
    for c in range(COLS + 1):
        draw.line([(c * CELL, y0), (c * CELL, y0 + ROWS * CELL)], fill=GRID)

    # food
    fr, fc = state["food"]
    draw.rectangle(
        [fc * CELL, y0 + fr * CELL, (fc + 1) * CELL - 1, y0 + (fr + 1) * CELL - 1],
        fill=FOOD
    )

    # snake
    snake = state["snake"]
    for i, (r, c) in enumerate(snake):
        color = SNAKE_HEAD if i == 0 else SNAKE_BODY
        draw.rectangle(
            [c * CELL, y0 + r * CELL, (c + 1) * CELL - 1, y0 + (r + 1) * CELL - 1],
            fill=color
        )

    # encode jpeg
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def ai_loop():
    global latest_jpeg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeEnv(rows=ROWS, cols=COLS)
    model = DQN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    state = env.reset()
    obs = get_observation(state)

    while True:
        # choose greedy action
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(torch.argmax(model(x), dim=1).item())

        abs_dir = apply_relative_action(state["dir"], action)
        state, reward, done, info = env.step(abs_dir)
        obs = get_observation(state)

        # restart on done (death or filled board)
        if done:
            state = env.reset()
            obs = get_observation(state)

        # update latest frame
        frame = render_frame(state)
        with latest_lock:
            latest_jpeg = frame

        time.sleep(1.0 / FPS)


@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=ai_loop, daemon=True)
    t.start()


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head>
        <title>Live AI Snake</title>
        <style>
          body { background:#111; color:#eee; font-family: Arial; display:flex; flex-direction:column; align-items:center; }
          h1 { margin-top: 20px; }
          img { margin-top: 10px; border: 2px solid #333; border-radius: 10px; }
          .note { margin-top: 10px; opacity: 0.8; }
        </style>
      </head>
      <body>
        <h1>Immortal Snake MWUHAHAHAH</h1>
        <img src="/stream" />
        <div class="note">This runs continuously on the server. Refresh anytime.</div>
      </body>
    </html>
    """


def mjpeg_generator():
    boundary = "frame"
    while True:
        with latest_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (b"--" + boundary.encode() + b"\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
               frame + b"\r\n")
        time.sleep(1.0 / FPS)


@app.get("/stream")
def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )