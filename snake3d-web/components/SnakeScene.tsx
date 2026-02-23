"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";
import type { Vec3 } from "../lib/snake3d";

export type CameraMode = "manual" | "follow" | "first" | "top" | "iso";

export type Frame = {
  size: Vec3;
  snake: Vec3[];
  food: Vec3;
  dir: number;
  score: number;
  done: boolean;
};

export default function SnakeScene({ frame, cameraMode }: { frame: Frame; cameraMode: CameraMode }) {
  return (
    <Canvas camera={{ position: [18, 14, 18], fov: 55, near: 0.1, far: 220 }} dpr={[1, 2]}>
      <color attach="background" args={["#07070a"]} />
      <fog attach="fog" args={["#07070a", 25, 95]} />

      <ambientLight intensity={0.65} />
      <directionalLight position={[12, 18, 10]} intensity={0.95} />
      <pointLight position={[-14, -10, -12]} intensity={0.35} />

      <SceneContent frame={frame} />
      <CameraRig frame={frame} mode={cameraMode} />

      {cameraMode === "manual" ? (
        <OrbitControls enablePan={false} minDistance={10} maxDistance={90} target={[0, 0, 0]} />
      ) : null}
    </Canvas>
  );
}

function CameraRig({ frame, mode }: { frame: Frame; mode: CameraMode }) {
  const { camera } = useThree();
  const [sx, sy, sz] = frame.size;

  const toWorld = (p: Vec3) =>
    new THREE.Vector3(p[0] - (sx - 1) / 2, p[1] - (sy - 1) / 2, p[2] - (sz - 1) / 2);

  useFrame(() => {
    const snake = frame.snake;
    const headG = snake[0];
    const neckG = snake[1] ?? headG;

    const head = toWorld(headG);
    const neck = toWorld(neckG);

    const forward = head.clone().sub(neck);
    if (forward.lengthSq() === 0) forward.set(1, 0, 0);
    forward.normalize();

    const up = new THREE.Vector3(0, 1, 0);
    let right = new THREE.Vector3().crossVectors(forward, up);
    if (right.lengthSq() === 0) right = new THREE.Vector3(0, 0, 1);
    right.normalize();

    const lookAt = head.clone();

    if (mode === "top") {
      const pos = head.clone().add(new THREE.Vector3(0, Math.max(sy, 10) * 1.6, 0));
      camera.position.lerp(pos, 0.12);
      camera.lookAt(lookAt.x, lookAt.y, lookAt.z);
      return;
    }

    if (mode === "iso") {
      const d = Math.max(sx, sy, sz) * 1.55;
      const pos = new THREE.Vector3(d, d * 0.9, d);
      camera.position.lerp(pos, 0.08);
      camera.lookAt(0, 0, 0);
      return;
    }

    if (mode === "first") {
      const pos = head.clone().add(up.clone().multiplyScalar(0.18)).add(forward.clone().multiplyScalar(0.18));
      const aim = head.clone().add(forward.clone().multiplyScalar(3.2));
      camera.position.lerp(pos, 0.25);
      camera.lookAt(aim.x, aim.y, aim.z);
      return;
    }

    if (mode === "follow") {
      const pos = head
        .clone()
        .add(forward.clone().multiplyScalar(-6.2))
        .add(up.clone().multiplyScalar(3.0))
        .add(right.clone().multiplyScalar(0.6));

      camera.position.lerp(pos, 0.10);
      camera.lookAt(lookAt.x, lookAt.y, lookAt.z);
      return;
    }
  });

  return null;
}

function SceneContent({ frame }: { frame: Frame }) {
  const [sx, sy, sz] = frame.size;

  const toWorld = (p: Vec3) =>
    new THREE.Vector3(p[0] - (sx - 1) / 2, p[1] - (sy - 1) / 2, p[2] - (sz - 1) / 2);

  const bodyGeom = useMemo(() => new THREE.BoxGeometry(0.92, 0.92, 0.92), []);
  const headGeom = useMemo(() => new THREE.SphereGeometry(0.62, 28, 28), []);
  const foodGeom = useMemo(() => new THREE.SphereGeometry(0.38, 24, 24), []);

  // Head color (change this anytime)
  const headColor = useMemo(() => new THREE.Color("#5DEBFF"), []); // neon cyan
  const midC = useMemo(() => new THREE.Color("#45d6aa"), []);
  const tailC = useMemo(() => new THREE.Color("#1f6b55"), []);

  const boundaryWire = useMemo(
    () => new THREE.MeshBasicMaterial({ color: "#ffffff", wireframe: true, transparent: true, opacity: 0.18 }),
    []
  );
  const boundarySolid = useMemo(
    () => new THREE.MeshStandardMaterial({ color: "#0d0d16", transparent: true, opacity: 0.10, roughness: 1.0 }),
    []
  );

  const foodMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: "#ff6b9a",
        emissive: "#ff6b9a",
        emissiveIntensity: 0.45,
        roughness: 0.35,
        metalness: 0.05,
      }),
    []
  );

  // grid lines on 3 faces
  const gridLinesXY = useMemo(() => makeGridLinesXY(sx, sy, sz), [sx, sy, sz]);
  const gridLinesXZ = useMemo(() => makeGridLinesXZ(sx, sy, sz), [sx, sy, sz]);
  const gridLinesYZ = useMemo(() => makeGridLinesYZ(sx, sy, sz), [sx, sy, sz]);

  const foodPos = toWorld(frame.food);

  return (
    <>
      <mesh>
        <boxGeometry args={[sx, sy, sz]} />
        <primitive object={boundarySolid} attach="material" />
      </mesh>
      <mesh>
        <boxGeometry args={[sx, sy, sz]} />
        <primitive object={boundaryWire} attach="material" />
      </mesh>

      <GridLines lines={gridLinesXY} opacity={0.08} />
      <GridLines lines={gridLinesXZ} opacity={0.08} />
      <GridLines lines={gridLinesYZ} opacity={0.08} />

      {frame.snake.map((seg, i) => {
        const p = toWorld(seg);

        if (i === 0) {
          const mat = new THREE.MeshStandardMaterial({
            color: headColor,
            emissive: headColor.clone(),
            emissiveIntensity: 0.55,
            roughness: 0.18,
            metalness: 0.08,
          });
          return <mesh key={`head-${i}`} geometry={headGeom} material={mat} position={[p.x, p.y, p.z]} />;
        }

        const t = frame.snake.length <= 2 ? 1 : (i - 1) / (frame.snake.length - 2);
        const c = new THREE.Color().lerpColors(midC, tailC, t);

        const mat = new THREE.MeshStandardMaterial({
          color: c,
          emissive: c.clone().multiplyScalar(0.35),
          emissiveIntensity: 0.14,
          roughness: 0.55,
          metalness: 0.03,
        });

        return (
          <mesh key={`${seg.join(",")}-${i}`} geometry={bodyGeom} material={mat} position={[p.x, p.y, p.z]} />
        );
      })}

      <mesh geometry={foodGeom} material={foodMat} position={[foodPos.x, foodPos.y, foodPos.z]} />
    </>
  );
}

function GridLines({ lines, opacity }: { lines: Array<[THREE.Vector3, THREE.Vector3]>; opacity: number }) {
  return (
    <>
      {lines.map(([a, b], i) => (
        <Line key={i} points={[a, b]} color="#ffffff" transparent opacity={opacity} lineWidth={1} />
      ))}
    </>
  );
}

function makeGridLinesXY(sx: number, sy: number, sz: number) {
  const z = -((sz - 1) / 2);
  const lines: Array<[THREE.Vector3, THREE.Vector3]> = [];
  for (let x = 0; x < sx; x++) {
    const X = x - (sx - 1) / 2;
    lines.push([new THREE.Vector3(X, -(sy - 1) / 2, z), new THREE.Vector3(X, (sy - 1) / 2, z)]);
  }
  for (let y = 0; y < sy; y++) {
    const Y = y - (sy - 1) / 2;
    lines.push([new THREE.Vector3(-(sx - 1) / 2, Y, z), new THREE.Vector3((sx - 1) / 2, Y, z)]);
  }
  return lines;
}
function makeGridLinesXZ(sx: number, sy: number, sz: number) {
  const y = -((sy - 1) / 2);
  const lines: Array<[THREE.Vector3, THREE.Vector3]> = [];
  for (let x = 0; x < sx; x++) {
    const X = x - (sx - 1) / 2;
    lines.push([new THREE.Vector3(X, y, -(sz - 1) / 2), new THREE.Vector3(X, y, (sz - 1) / 2)]);
  }
  for (let z = 0; z < sz; z++) {
    const Z = z - (sz - 1) / 2;
    lines.push([new THREE.Vector3(-(sx - 1) / 2, y, Z), new THREE.Vector3((sx - 1) / 2, y, Z)]);
  }
  return lines;
}
function makeGridLinesYZ(sx: number, sy: number, sz: number) {
  const x = -((sx - 1) / 2);
  const lines: Array<[THREE.Vector3, THREE.Vector3]> = [];
  for (let y = 0; y < sy; y++) {
    const Y = y - (sy - 1) / 2;
    lines.push([new THREE.Vector3(x, Y, -(sz - 1) / 2), new THREE.Vector3(x, Y, (sz - 1) / 2)]);
  }
  for (let z = 0; z < sz; z++) {
    const Z = z - (sz - 1) / 2;
    lines.push([new THREE.Vector3(x, -(sy - 1) / 2, Z), new THREE.Vector3(x, (sy - 1) / 2, Z)]);
  }
  return lines;
}