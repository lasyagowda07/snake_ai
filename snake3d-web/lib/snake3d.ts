export type Vec3 = [number, number, number];

export const DIRS: Vec3[] = [
  [ 1, 0, 0], // +X (0)
  [-1, 0, 0], // -X (1)
  [ 0, 1, 0], // +Y (2)
  [ 0,-1, 0], // -Y (3)
  [ 0, 0, 1], // +Z (4)
  [ 0, 0,-1], // -Z (5)
];
export const OPP: Record<number, number> = { 0:1, 1:0, 2:3, 3:2, 4:5, 5:4 };

export type Snake3DState = {
  size: Vec3;
  snake: Vec3[]; // head at 0
  food: Vec3;
  dir: number;
  score: number;
  done: boolean;
};

function key(p: Vec3) { return `${p[0]},${p[1]},${p[2]}`; }

export function reset3D(size: Vec3): Snake3DState {
  const [sx, sy, sz] = size;
  const cx = Math.floor(sx/2), cy = Math.floor(sy/2), cz = Math.floor(sz/2);
  const snake: Vec3[] = [[cx,cy,cz],[cx-1,cy,cz],[cx-2,cy,cz]];
  const dir = 0;
  const food = placeFood(size, snake);
  return { size, snake, food, dir, score: 0, done: false };
}

function inBounds(size: Vec3, p: Vec3) {
  const [sx,sy,sz] = size;
  return p[0]>=0 && p[0]<sx && p[1]>=0 && p[1]<sy && p[2]>=0 && p[2]<sz;
}

function placeFood(size: Vec3, snake: Vec3[]): Vec3 {
  const [sx,sy,sz] = size;
  const s = new Set(snake.map(key));
  const empties: Vec3[] = [];
  for (let x=0;x<sx;x++) for (let y=0;y<sy;y++) for (let z=0;z<sz;z++) {
    const p: Vec3 = [x,y,z];
    if (!s.has(key(p))) empties.push(p);
  }
  if (empties.length === 0) return [0,0,0]; // will be handled as "filled"
  return empties[(Math.random()*empties.length) | 0];
}

export function step3D(state: Snake3DState, actionDir: number): Snake3DState {
  if (state.done) return state;

  let dir = state.dir;
  if (actionDir >= 0 && actionDir <= 5 && actionDir !== OPP[dir]) dir = actionDir;

  const [dx,dy,dz] = DIRS[dir];
  const head = state.snake[0];
  const nh: Vec3 = [head[0]+dx, head[1]+dy, head[2]+dz];

  if (!inBounds(state.size, nh)) {
    return { ...state, dir, done: true };
  }

  const snakeSet = new Set(state.snake.map(key));
  const tail = state.snake[state.snake.length - 1];
  const willGrow = (nh[0]===state.food[0] && nh[1]===state.food[1] && nh[2]===state.food[2]);

  if (snakeSet.has(key(nh))) {
    const isTail = key(nh) === key(tail);
    if (!(isTail && !willGrow)) {
      return { ...state, dir, done: true };
    }
  }

  const newSnake = [nh, ...state.snake];

  let score = state.score;
  let food = state.food;

  if (willGrow) {
    score += 1;
    food = placeFood(state.size, newSnake);
  } else {
    newSnake.pop();
  }

  // filled board detection: if food lands on snake set repeatedly could be avoided;
  // here: if snake length == volume => done.
  const [sx,sy,sz] = state.size;
  const volume = sx*sy*sz;
  const done = newSnake.length >= volume;

  return { size: state.size, snake: newSnake, food, dir, score, done };
}

// 18-d observation to match your PyTorch trainer
export function obs18(state: Snake3DState): Float32Array {
  const { size, snake, food, dir } = state;
  const head = snake[0];

  // collision check for candidate cell
  const snakeSet = new Set(snake.map(key));
  const tail = snake[snake.length - 1];

  const willGrowAt = (p: Vec3) =>
    p[0]===food[0] && p[1]===food[1] && p[2]===food[2];

  const isCollision = (p: Vec3) => {
    if (!inBounds(size, p)) return 1;
    const k = key(p);
    if (!snakeSet.has(k)) return 0;
    // tail is safe if not growing
    const isTail = k === key(tail);
    if (isTail && !willGrowAt(p)) return 0;
    return 1;
  };

  const danger = new Array(6).fill(0).map((_, d) => {
    const [dx,dy,dz] = DIRS[d];
    const p: Vec3 = [head[0]+dx, head[1]+dy, head[2]+dz];
    return isCollision(p);
  });

  const dirOH = new Array(6).fill(0).map((_, d) => (d === dir ? 1 : 0));

  const foodRel = [
    food[0] > head[0] ? 1 : 0, // +x
    food[0] < head[0] ? 1 : 0, // -x
    food[1] > head[1] ? 1 : 0, // +y
    food[1] < head[1] ? 1 : 0, // -y
    food[2] > head[2] ? 1 : 0, // +z
    food[2] < head[2] ? 1 : 0, // -z
  ];

  return new Float32Array([...danger, ...dirOH, ...foodRel]);
}