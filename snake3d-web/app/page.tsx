"use client";

import { useEffect, useRef, useState } from "react";
import SnakeScene, { type Frame, type CameraMode } from "../components/SnakeScene";
import { loadPolicy, chooseAction } from "../lib/onnxPolicy";
import { obs18, reset3D, step3D } from "../lib/snake3d";

export default function Page() {
  const [mode, setMode] = useState<CameraMode>("manual");
  const [frame, setFrame] = useState<Frame>(() => reset3D([10, 10, 10]));
  const sessionRef = useRef<any>(null);
  const runningRef = useRef(true);

  // load model once
  useEffect(() => {
    (async () => {
      sessionRef.current = await loadPolicy("/snake_dqn3d.onnx");
    })();
    return () => {
      runningRef.current = false;
    };
  }, []);

  // main loop
  useEffect(() => {
    let raf = 0;
    let last = performance.now();

    const tick = async (now: number) => {
      raf = requestAnimationFrame(tick);

      // cap to ~12 FPS sim so ONNX isn't too heavy
      if (now - last < 80) return;
      last = now;

      const session = sessionRef.current;
      if (!session) return;

      setFrame((prev) => {
        // auto reset on done or filled
        if (prev.done) return reset3D(prev.size);

        const obs = obs18(prev);
        // fire and forget; we’ll apply action next frame via state closure
        // BUT we need sync. So we do a small trick: block with a promise in an IIFE.
        return prev;
      });

      // We need to compute action using the latest frame.
      // To keep it simple, compute from current state ref.
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Better loop: keep latest state in ref so inference can be awaited
  const stateRef = useRef(frame);
  useEffect(() => { stateRef.current = frame; }, [frame]);

  useEffect(() => {
    let alive = true;

    const loop = async () => {
      while (alive && runningRef.current) {
        const session = sessionRef.current;
        if (!session) {
          await sleep(50);
          continue;
        }

        const s = stateRef.current;
        const obs = obs18(s);
        const action = await chooseAction(session, obs);
        const next = step3D(s, action);

        // reset automatically
        const finalState = next.done ? reset3D(next.size) : next;
        stateRef.current = finalState;
        setFrame(finalState);

        await sleep(80); // ~12.5 FPS
      }
    };

    loop();
    return () => { alive = false; };
  }, []);

  return (
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "26px 18px 38px", color: "#f3f3f6" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap", alignItems: "end" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 34, lineHeight: 1.05, letterSpacing: "-0.02em" }}>
            immortal snake <span style={{ opacity: 0.7 }}>(^.^ mwuhahaha)</span>
          </h1>
          <p style={{ marginTop: 10, opacity: 0.75, fontSize: 13 }}>
            The AI-cursed snake is stuck in a loop for eternity. Auto-resets on death or full grid.
          </p>
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <Pill label={`score: ${frame.score}`} />
          <Pill label={`size: ${frame.size[0]}×${frame.size[1]}×${frame.size[2]}`} />
        </div>
      </div>

      <div style={{ marginTop: 14, display: "flex", gap: 10, flexWrap: "wrap" }}>
        <ModeButton label="manual" active={mode === "manual"} onClick={() => setMode("manual")} />
        <ModeButton label="follow" active={mode === "follow"} onClick={() => setMode("follow")} />
        {/* <ModeButton label="first person" active={mode === "first"} onClick={() => setMode("first")} /> */}
        <ModeButton label="top-down" active={mode === "top"} onClick={() => setMode("top")} />
        <ModeButton label="isometric" active={mode === "iso"} onClick={() => setMode("iso")} />
      </div>

      <div style={{
        marginTop: 18, height: "74vh", minHeight: 520,
        borderRadius: 22, overflow: "hidden",
        background: "rgba(10,10,14,0.55)",
        border: "1px solid rgba(255,255,255,0.07)",
        boxShadow: "0 30px 90px rgba(0,0,0,0.55)"
      }}>
        <SnakeScene frame={frame} cameraMode={mode} />
      </div>
    </div>
  );
}

function Pill({ label }: { label: string }) {
  return (
    <div style={{
      padding: "8px 12px", borderRadius: 999,
      background: "rgba(18,18,28,0.62)",
      border: "1px solid rgba(255,255,255,0.08)",
      boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
      fontSize: 13
    }}>
      {label}
    </div>
  );
}

function ModeButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "10px 14px",
        borderRadius: 999,
        border: active ? "1px solid rgba(93,235,255,0.55)" : "1px solid rgba(255,255,255,0.10)",
        background: active ? "rgba(93,235,255,0.10)" : "rgba(18,18,28,0.60)",
        color: "rgba(243,243,246,0.92)",
        boxShadow: active ? "0 10px 30px rgba(93,235,255,0.10)" : "0 10px 30px rgba(0,0,0,0.35)",
        cursor: "pointer",
        fontSize: 13,
        backdropFilter: "blur(12px)",
      }}
    >
      {label}
    </button>
  );
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}