import * as ort from "onnxruntime-web";

export async function loadPolicy(modelUrl = "/snake_dqn3d.onnx") {
  // wasm backend is default; this works on Vercel static
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
  });
  return session;
}

export async function chooseAction(session: ort.InferenceSession, obs: Float32Array) {
  const input = new ort.Tensor("float32", obs, [1, obs.length]);
  const outputs = await session.run({ obs: input });
  const q = outputs.q.data as Float32Array;

  // argmax
  let best = 0;
  for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
  return best;
}