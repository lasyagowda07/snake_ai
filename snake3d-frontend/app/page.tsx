"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";

type Frame = {
  size: [number, number, number];
  snake: [number, number, number][];
  food: [number, number, number];
  dir: number;
  score: number;
};

type CameraMode = "manual" | "follow" | "first" | "top" | "iso";

const SnakeScene = dynamic(() => import("../components/SnakeScene"), { ssr: false });

export default function Page() {
  const wsUrl = process.env.NEXT_PUBLIC_SNAKE_WS_URL || "ws://127.0.0.1:8000/ws";

  const [frame, setFrame] = useState<Frame | null>(null);
  const [status, setStatus] = useState<"connecting" | "open" | "closed" | "error">("connecting");
  const [cameraMode, setCameraMode] = useState<CameraMode>("iso");

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;

    const connect = () => {
      if (!alive) return;
      setStatus("connecting");

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => alive && setStatus("open");
      ws.onmessage = (evt) => {
        if (!alive) return;
        try {
          setFrame(JSON.parse(evt.data));
        } catch {}
      };
      ws.onerror = () => alive && setStatus("error");
      ws.onclose = () => {
        if (!alive) return;
        setStatus("closed");
        if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = window.setTimeout(connect, 900);
      };
    };

    connect();
    return () => {
      alive = false;
      if (reconnectTimerRef.current) window.clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [wsUrl]);

  return (
    <div className="container">
      <motion.div
        className="header"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: "easeOut" }}
      >
        <div>
          <h1 className="title">Immortal Snake <span style={{ opacity: 0.7 }}>(Mwuhahaha)</span></h1>
          <p className="sub">
            WebSocket: <span style={{ color: "rgba(243,243,246,0.92)" }}>{wsUrl}</span>
          </p>
        </div>

        <div className="badges">
          <div className="badge">status: <b>{status}</b></div>
          <div className="badge">score: <b>{frame?.score ?? "-"}</b></div>
          <div className="badge">
            size: <b>{frame ? `${frame.size[0]}×${frame.size[1]}×${frame.size[2]}` : "-"}</b>
          </div>
        </div>
      </motion.div>

      {/* Camera mode buttons */}
      <div style={{ marginTop: 14, display: "flex", gap: 10, flexWrap: "wrap" }}>
        <ModeButton label="manual" active={cameraMode === "manual"} onClick={() => setCameraMode("manual")} />
        <ModeButton label="follow" active={cameraMode === "follow"} onClick={() => setCameraMode("follow")} />
        {/* <ModeButton label="first person" active={cameraMode === "first"} onClick={() => setCameraMode("first")} /> */}
        <ModeButton label="top-down" active={cameraMode === "top"} onClick={() => setCameraMode("top")} />
        <ModeButton label="isometric" active={cameraMode === "iso"} onClick={() => setCameraMode("iso")} />
      </div>

      <motion.div
        className="stage"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.55, delay: 0.08 }}
      >
        <SnakeScene frame={frame} cameraMode={cameraMode} />
      </motion.div>

      <div className="footerNote">
      This is a live 3D reinforcement learning demo. The snake plays endlessly. When it dies or fills the space, the world resets.
      </div>
      
    </div>
  );
}

function ModeButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "10px 14px",
        borderRadius: 999,
        border: active ? "1px solid rgba(255,184,107,0.55)" : "1px solid rgba(255,255,255,0.10)",
        background: active ? "rgba(255,184,107,0.12)" : "rgba(18,18,28,0.60)",
        color: "rgba(243,243,246,0.92)",
        boxShadow: active ? "0 10px 30px rgba(255,184,107,0.12)" : "0 10px 30px rgba(0,0,0,0.35)",
        cursor: "pointer",
        fontSize: 13,
        backdropFilter: "blur(12px)",
      }}
    >
      {label}
    </button>
  );
}