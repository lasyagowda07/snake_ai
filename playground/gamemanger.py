import time
from physics import World, Particle
import display


def main():
    # -----------------------
    # CONFIG
    # -----------------------
    rows = 40
    cols = 40
    empty_cell = " "

    gravity = 1.0
    dt = 1.0
    sleep_seconds = 1.0
    max_steps = 40

    # OBSERVE FLAG (trail on/off)
    observe = True   # set to False to show only current positions

    # -----------------------
    # WORLD
    # -----------------------
    world = World(rows=rows, cols=cols, empty=empty_cell)

    # Add particles
    world.add_particle(Particle(x=0, y=0, vx=1, vy=1, ax=1, ay=1, symbol="A"))
    world.add_particle(Particle(x=1, y=0, vx=1, vy=1, ax=1, ay=1, symbol="B"))
    world.add_particle(Particle(x=0, y=3, vx=1, vy=1, ax=1, ay=1, symbol="C"))
    # -----------------------
    # LOOP
    # -----------------------
    step = 0
    while step < max_steps and world.alive_particles() > 0:
        world.step(dt=dt, gravity=gravity, kill_out_of_bounds=True)

        grid = world.build_grid(observe=observe, trail_symbol=".")
        print(f"\nStep: {step} | Alive: {world.alive_particles()} | observe={observe}")
        display.world_view(grid, show_coords=True)

        step += 1
        time.sleep(sleep_seconds)

    print("\nGame over")


if __name__ == "__main__":
    main()