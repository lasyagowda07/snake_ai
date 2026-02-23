import os
import torch
from rl3d.model3d import DQN3D

MODEL_PATH = "rl3d/checkpoints/snake3d_dqn.pt"
OUT_PATH = "snake_dqn3d.onnx"

def main():
    device = "cpu"
    model = DQN3D(input_dim=18, hidden=256, output_dim=6).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dummy = torch.randn(1, 18, device=device)

    torch.onnx.export(
        model,
        dummy,
        OUT_PATH,
        input_names=["obs"],
        output_names=["q"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "q": {0: "batch"}},
    )
    print("exported:", OUT_PATH)

if __name__ == "__main__":
    main()