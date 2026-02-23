import pygame
from core.snake_env import SnakeEnv

CELL = 24
FPS = 12

KEY_TO_DIR = {
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_RIGHT: "RIGHT",
}

def main():
    env = SnakeEnv(rows=20, cols=20)
    state = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((state["cols"] * CELL, state["rows"] * CELL))
    pygame.display.set_caption("Snake (Stage 1)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    action = state["dir"]
    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_TO_DIR:
                    action = KEY_TO_DIR[event.key]
                if event.key == pygame.K_r:
                    state = env.reset()
                    action = state["dir"]

        state, reward, done, info = env.step(action)
        if done:
            # pause and reset quickly
            state = env.reset()
            action = state["dir"]

        # draw
        screen.fill((20, 20, 20))

        # food
        fr, fc = state["food"]
        pygame.draw.rect(screen, (255, 80, 80), (fc * CELL, fr * CELL, CELL, CELL))

        # snake
        for idx, (r, c) in enumerate(state["snake"]):
            color = (80, 220, 120) if idx == 0 else (60, 170, 100)
            pygame.draw.rect(screen, color, (c * CELL, r * CELL, CELL, CELL))

        text = font.render(f"score: {state['score']}  (R to reset)", True, (240, 240, 240))
        screen.blit(text, (8, 8))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()