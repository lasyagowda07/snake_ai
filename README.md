🐍 3D AI Snake — Browser-Based Reinforcement Learning

From Gravity Simulation to a Self-Learning Snake

This project didn’t start as a game.

It began with a simple physics experiment — simulating gravity and parabolic projectile motion on a 2D grid. I built a small world, added particles, applied acceleration, and watched trajectories evolve step by step. That exploration of motion, simulation loops, and state updates led to a bigger question:

What if instead of simulating physics… I simulate intelligence?

That idea evolved into building a Snake game — and then training an AI to master it.

⸻

🎮 Phase 1 — Building the Snake Environment

The first step was creating the game logic itself.

Instead of rendering first, I built the environment as a pure simulation layer:
	•	3D grid world (e.g., 10×10×10)
	•	Snake represented as an ordered list of coordinates
	•	Food spawning in empty cells
	•	Collision detection (walls + self)
	•	Growth mechanics
	•	Automatic reset on death or full grid

The environment is fully deterministic and independent of rendering.

It exposes:
	•	Current state
	•	Action step function
	•	Observation vector (18 features)
	•	Reward logic
	•	Done flag

This separation of simulation from visualization made it easy to plug AI into it later.

⸻

🤖 Phase 2 — Training the AI (Deep Q-Learning)

Once the environment worked, I trained a reinforcement learning agent using Deep Q-Network (DQN).

Model architecture:
	•	Input: 18-dimensional state vector
	•	Hidden layer: 256 units
	•	Output: 6 actions (±X, ±Y, ±Z directions)

Training components:
	•	Experience replay buffer
	•	Epsilon-greedy exploration
	•	Reward shaping
	•	Target Q-learning updates

Over ~1500 episodes, the model learned:
	•	To avoid collisions
	•	To navigate efficiently
	•	To pursue food strategically
	•	To survive longer

The trained model was saved as a PyTorch .pt checkpoint.

⸻

🌍 Phase 3 — Making It Run in the Browser (No Backend)

Instead of hosting a backend server, I converted the trained PyTorch model into ONNX format.

Why ONNX?
	•	Portable
	•	Browser compatible
	•	Works with onnxruntime-web
	•	Removes the need for EC2 or WebSockets

Now the entire AI runs client-side.

No server.
No cloud compute.
Fully free deployment.

⸻

🎨 Phase 4 — 3D Visualization

The frontend was built using:
	•	Next.js (App Router)
	•	React Three Fiber
	•	Drei
	•	WebGL

Features include:
	•	Dark cinematic 3D environment
	•	Boundary cube + multi-plane grid system
	•	Gradient snake body
	•	Glowing round AI head
	•	Dynamic food rendering
	•	Multiple camera modes:
	•	Manual (Orbit)
	•	Follow (3rd person)
	•	First person
	•	Top-down
	•	Isometric

The snake runs in a continuous loop:
	•	Observe → Infer → Act → Render → Repeat
	•	Auto-reset on death
	•	Auto-reset on full grid

Hence the tagline:

The AI-cursed snake is stuck in a loop for eternity.

⸻

🧠 Architecture Overview

Browser Flow:
	1.	Load ONNX model
	2.	Initialize snake state
	3.	Every ~80ms:
	•	Generate observation
	•	Run ONNX inference
	•	Select action (argmax Q-value)
	•	Step environment
	4.	Render updated state in 3D
	5.	Reset when game ends
	6.	Repeat forever

All inference runs locally in the user’s browser.

⸻

🚀 Deployment

Deployment is fully static:
	•	Push to GitHub
	•	Import into Vercel
	•	Deploy

No backend.
No server.
No infrastructure cost.

The AI lives entirely in the browser.

⸻

🔬 What This Project Demonstrates
	•	Physics-based simulation foundations
	•	Environment design for reinforcement learning
	•	Deep Q-Network implementation
	•	Model export to ONNX
	•	Client-side inference
	•	3D rendering with WebGL
	•	Clean architecture separation (simulation / model / rendering)

⸻

This project evolved from simulating gravity to simulating intelligence — and eventually to visualizing a self-learning agent navigating a 3D world in real time.