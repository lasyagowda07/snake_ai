def world_view(grid: list[list[str]], show_coords: bool = True) -> None:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if show_coords:
        header = "    " + " ".join(f"{c:>2}" for c in range(cols))
        print(header)

    print("   +" + "---" * cols + "+")

    for r in range(rows):
        row_str = " ".join(f"{grid[r][c]:>2}" for c in range(cols))
        if show_coords:
            print(f"{r:>2} | {row_str} |")
        else:
            print(f"| {row_str} |")

    print("   +" + "---" * cols + "+")