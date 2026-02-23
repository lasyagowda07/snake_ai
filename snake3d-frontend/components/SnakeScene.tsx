"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

type Frame = {
  size: [number, number, number];
  snake: [number, number, number][];
  food: [number, number, number];
  dir: number;
  score: number;
};

type CameraMode = "manual" | "follow" | "first" | "top" | "iso";

export default function SnakeScene({
  frame,
  cameraMode,
}: {
  frame: Frame | null;
  cameraMode: CameraMode;
}) {
  const size: [number, number, number] = frame?.size ?? [10, 10, 10];
  const snake = frame?.snake ?? [[5, 5, 5]];
  const food = frame?.food ?? [7, 5, 5];

  return (
    <Canvas
      camera={{ position: [18, 14, 18], fov: 55, near: 0.1, far: 220 }}
      dpr={[1, 2]}
    >
      <color attach="background" args={["#07070a"]} />
      <fog attach="fog" args={["#07070a", 25, 95]} />

      <ambientLight intensity={0.65} />
      <directionalLight position={[12, 18, 10]} intensity={0.95} />
      <pointLight position={[-14, -10, -12]} intensity={0.35} />
      <pointLight position={[0, 18, 0]} intensity={0.25} />

      <SceneContent size={size} snake={snake} food={food} />

      <CameraRig size={size} snake={snake} mode={cameraMode} />

      {/* Orbit only in manual mode */}
      {cameraMode === "manual" ? (
        <OrbitControls enablePan={false} minDistance={10} maxDistance={90} target={[0, 0, 0]} />
      ) : null}
    </Canvas>
  );
}

function CameraRig({
  size,
  snake,
  mode,
}: {
  size: [number, number, number];
  snake: [number, number, number][];
  mode: CameraMode;
}) {
  const [sx, sy, sz] = size;
  const { camera } = useThree();

  const toWorld = (p: [number, number, number]) => {
    const [x, y, z] = p;
    return new THREE.Vector3(x - (sx - 1) / 2, y - (sy - 1) / 2, z - (sz - 1) / 2);
  };

  useFrame(() => {
    if (!snake?.length) return;

    const headG = snake[0];
    const neckG = snake[1] ?? headG;

    const head = toWorld(headG);
    const neck = toWorld(neckG);

    const forward = head.clone().sub(neck);
    if (forward.lengthSq() === 0) forward.set(1, 0, 0);
    forward.normalize();

    // pick a stable up/right
    const up = new THREE.Vector3(0, 1, 0);
    let right = new THREE.Vector3().crossVectors(forward, up);
    if (right.lengthSq() === 0) right = new THREE.Vector3(0, 0, 1);
    right.normalize();

    // Targets
    const lookAt = head.clone();

    if (mode === "top") {
      const pos = head.clone().add(new THREE.Vector3(0, Math.max(sy, 10) * 1.6, 0));
      camera.position.lerp(pos, 0.12);
      camera.lookAt(lookAt.x, lookAt.y, lookAt.z);
      return;
    }

    if (mode === "iso") {
      const d = Math.max(sx, sy, sz) * 1.5;
      const pos = new THREE.Vector3(d, d * 0.9, d);
      camera.position.lerp(pos, 0.08);
      camera.lookAt(0, 0, 0);
      return;
    }

    if (mode === "first") {
      // camera at head, looking forward
      const pos = head
        .clone()
        .add(up.clone().multiplyScalar(0.15))
        .add(forward.clone().multiplyScalar(0.15));
      const aim = head.clone().add(forward.clone().multiplyScalar(3.0));

      camera.position.lerp(pos, 0.25);
      camera.lookAt(aim.x, aim.y, aim.z);
      return;
    }

    if (mode === "follow") {
      // 3rd person follow: behind and above head
      const back = forward.clone().multiplyScalar(-1);
      const height = up.clone().multiplyScalar(3.0);
      const side = right.clone().multiplyScalar(0.6);
      const pos = head.clone().add(back.multiplyScalar(6.2)).add(height).add(side);

      camera.position.lerp(pos, 0.10);
      camera.lookAt(lookAt.x, lookAt.y, lookAt.z);
      return;
    }

    // manual: do nothing here (OrbitControls)
  });

  return null;
}

function SceneContent({
  size,
  snake,
  food,
}: {
  size: [number, number, number];
  snake: [number, number, number][];
  food: [number, number, number];
}) {
  const [sx, sy, sz] = size;

  const toWorld = (p: [number, number, number]) => {
    const [x, y, z] = p;
    return new THREE.Vector3(x - (sx - 1) / 2, y - (sy - 1) / 2, z - (sz - 1) / 2);
  };

  const bodyGeom = useMemo(() => new THREE.BoxGeometry(0.92, 0.92, 0.92), []);
  const headGeom = useMemo(() => new THREE.SphereGeometry(0.62, 28, 28), []);
  const foodGeom = useMemo(() => new THREE.SphereGeometry(0.38, 24, 24), []);

  const headC = useMemo(() => new THREE.Color("#5DEBFF"), []); // warm head
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

  const gridColor = "#ffffff";
  const gridOpacity = 0.08;
  const gridLinesXY = useMemo(() => makeGridLinesXY(sx, sy, sz), [sx, sy, sz]);
  const gridLinesXZ = useMemo(() => makeGridLinesXZ(sx, sy, sz), [sx, sy, sz]);
  const gridLinesYZ = useMemo(() => makeGridLinesYZ(sx, sy, sz), [sx, sy, sz]);

  const foodPos = toWorld(food);

  return (
    <>
      {/* Boundary */}
      <mesh>
        <boxGeometry args={[sx, sy, sz]} />
        <primitive object={boundarySolid} attach="material" />
      </mesh>
      <mesh>
        <boxGeometry args={[sx, sy, sz]} />
        <primitive object={boundaryWire} attach="material" />
      </mesh>

      {/* Grid planes */}
      <group>
        <GridLines lines={gridLinesXY} color={gridColor} opacity={gridOpacity} />
        <GridLines lines={gridLinesXZ} color={gridColor} opacity={gridOpacity} />
        <GridLines lines={gridLinesYZ} color={gridColor} opacity={gridOpacity} />
      </group>

      {/* Snake */}
      {snake.map((seg, i) => {
        const p = toWorld(seg);

        // head: round, warm color, glow
        if (i === 0) {
          const headMat = new THREE.MeshStandardMaterial({
            color: headC,
            emissive: headC.clone().multiplyScalar(0.7),
            emissiveIntensity: 0.40,
            roughness: 0.25,
            metalness: 0.05,
          });

          return (
            <mesh
              key={`head-${seg[0]}-${seg[1]}-${seg[2]}`}
              geometry={headGeom}
              material={headMat}
              position={[p.x, p.y, p.z]}
            />
          );
        }

        // body gradient
        const t = snake.length <= 2 ? 1 : (i - 1) / (snake.length - 2);
        const c = new THREE.Color();
        c.lerpColors(midC, tailC, t);

        const mat = new THREE.MeshStandardMaterial({
          color: c,
          emissive: c.clone().multiplyScalar(0.35),
          emissiveIntensity: 0.14,
          roughness: 0.55,
          metalness: 0.03,
        });

        return (
          <mesh
            key={`${seg[0]}-${seg[1]}-${seg[2]}-${i}`}
            geometry={bodyGeom}
            material={mat}
            position={[p.x, p.y, p.z]}
          />
        );
      })}

      {/* Food */}
      <mesh geometry={foodGeom} material={foodMat} position={[foodPos.x, foodPos.y, foodPos.z]} />
    </>
  );
}

function GridLines({
  lines,
  color,
  opacity,
}: {
  lines: Array<[THREE.Vector3, THREE.Vector3]>;
  color: string;
  opacity: number;
}) {
  return (
    <>
      {lines.map(([a, b], i) => (
        <Line key={i} points={[a, b]} color={color} transparent opacity={opacity} lineWidth={1} />
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