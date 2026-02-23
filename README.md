3D AI Snake

Reinforcement Learning in the Browser

⸻

Overview

This project began as a simple physics experiment.

It started with simulating gravity and parabolic motion on a 2D grid — applying acceleration, updating state over time, and observing projectile trajectories.

From simulating motion, the next natural question emerged:

What if instead of simulating physics, I simulate intelligence?

That curiosity led to building a Snake game — and eventually training an AI to master it in three dimensions.

⸻

Project Evolution

⸻

1. Physics Simulation

The initial phase focused on building a discrete world simulation:
	•	Grid-based environment
	•	Acceleration and gravity modeling
	•	Parabolic motion updates
	•	Deterministic step-based world transitions

This phase established the architectural foundation:
	•	World state representation
	•	Update loop
	•	Separation between simulation and visualization

That separation became critical later when introducing reinforcement learning.

⸻

2. Building the 3D Snake Environment

Before introducing AI, the environment was built independently.

Core components:
	•	Configurable 3D grid (e.g., 10 × 10 × 10)
	•	Snake stored as an ordered list of coordinates (head-first)
	•	Food spawning only in empty cells
	•	Collision detection:
	•	Wall collision
	•	Self-collision
	•	Growth mechanics
	•	Automatic reset on death or full-grid completion

The environment exposes:
	•	step(action)
	•	state
	•	done flag
	•	reward system
	•	18-dimensional observation vector

The simulation layer is completely independent from rendering.
This allowed reuse for both training and browser inference.

⸻

3. Training the AI (Deep Q-Network)

The agent was trained using Deep Q-Learning.

Model architecture:
	•	Input: 18-dimensional state vector
	•	Hidden layer: 256 units
	•	Output: 6 movement actions (±X, ±Y, ±Z)

Training components:
	•	Experience replay buffer
	•	Epsilon-greedy exploration
	•	Reward shaping
	•	Target Q-learning

After ~1500 episodes, the agent learned to:
	•	Avoid collisions
	•	Navigate toward food efficiently
	•	Survive longer
	•	Optimize movement in 3D space

The trained model was saved as a PyTorch checkpoint.

⸻

4. Converting to ONNX

To remove backend dependency, the model was exported to ONNX format.

This enabled:
	•	Browser-based inference
	•	No server required
	•	No WebSocket architecture
	•	Fully static deployment

The ONNX model is loaded in the browser using onnxruntime-web.

⸻

5. 3D Visualization

Frontend stack:
	•	Next.js (App Router)
	•	React Three Fiber
	•	Drei
	•	WebGL

Rendering features:
	•	Dark cinematic 3D environment
	•	Boundary cube visualization
	•	Multi-plane grid system
	•	Gradient snake body
	•	Distinct glowing spherical head
	•	Dynamic food rendering
	•	Multiple camera modes:
	•	Manual (Orbit)
	•	Follow (third person)
	•	First person
	•	Top-down
	•	Isometric

Rendering is fully decoupled from simulation logic.

⸻

Runtime Architecture

When the page loads:
	1.	The ONNX model loads in the browser.
	2.	The snake environment initializes.
	3.	A loop runs at fixed intervals:
	•	Generate observation
	•	Run ONNX inference
	•	Select action (argmax Q-values)
	•	Step environment
	•	Re-render scene
	4.	On collision or full-grid completion:
	•	Environment resets automatically

All inference runs client-side.

There is no backend server.

⸻

Deployment

The project is deployed as a fully static application:
	•	Source hosted on GitHub
	•	Deployed via Vercel
	•	No backend infrastructure
	•	No cloud compute
	•	No server maintenance

The AI runs entirely in the user’s browser.

⸻

Technical Concepts Demonstrated
	•	Physics-based simulation foundations
	•	Discrete environment design
	•	Reinforcement learning (Deep Q-Network)
	•	Model export to ONNX
	•	Client-side neural network inference
	•	3D rendering with WebGL
	•	Clean architecture separation:
	•	Simulation
	•	Model
	•	Rendering

⸻

Closing

This project evolved from simulating gravity to simulating intelligence — and ultimately to visualizing a self-learning agent navigating a 3D world in real time.