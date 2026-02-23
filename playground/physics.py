from dataclasses import dataclass, field
import math


@dataclass
class Particle:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    symbol: str = "X"
    alive: bool = True

    # stores visited cells as (row, col)
    trail: set[tuple[int, int]] = field(default_factory=set)

    def _cell(self) -> tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def step(self, dt: float, gravity: float) -> None:
        if not self.alive:
            return

        # update velocities
        self.vx += self.ax * dt
        self.vy += (self.ay - gravity) * dt

        # update positions
        self.x += self.vx * dt
        self.y += self.vy * dt


class World:
    def __init__(self, rows: int, cols: int, empty: str = "_") -> None:
        self.rows = rows
        self.cols = cols
        self.empty = empty
        self.particles: list[Particle] = []

    def add_particle(self, particle: Particle) -> None:
        # record starting cell in trail too (useful for observe mode)
        r, c = int(round(particle.x)), int(round(particle.y))
        particle.trail.add((r, c))
        self.particles.append(particle)

    def in_bounds_cell(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def in_bounds(self, x: float, y: float) -> bool:
        r = int(round(x))
        c = int(round(y))
        return self.in_bounds_cell(r, c)

    def step(self, dt: float, gravity: float, kill_out_of_bounds: bool = True) -> None:
        for p in self.particles:
            if not p.alive:
                continue

            p.step(dt, gravity)

            r, c = p._cell()
            if kill_out_of_bounds and not self.in_bounds_cell(r, c):
                p.alive = False
                continue

            # record visited cell
            p.trail.add((r, c))

    def build_grid(self, observe: bool = False, trail_symbol: str = ".") -> list[list[str]]:
        """
        observe=False  -> only current positions
        observe=True   -> draw trail + current position on top
        """
        grid = [[self.empty for _ in range(self.cols)] for _ in range(self.rows)]

        if observe:
            # draw trails first
            for p in self.particles:
                for (r, c) in p.trail:
                    if self.in_bounds_cell(r, c) and grid[r][c] == self.empty:
                        grid[r][c] = trail_symbol

        # draw current positions on top
        for p in self.particles:
            if not p.alive:
                continue
            r, c = int(round(p.x)), int(round(p.y))
            if self.in_bounds_cell(r, c):
                grid[r][c] = p.symbol

        return grid

    def alive_particles(self) -> int:
        return sum(1 for p in self.particles if p.alive)