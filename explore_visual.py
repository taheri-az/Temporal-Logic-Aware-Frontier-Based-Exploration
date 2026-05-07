import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def manhattan_dist(c1, c2, m):
    r1, c1 = divmod(c1, m)
    r2, c2 = divmod(c2, m)
    return abs(r1 - r2) + abs(c1 - c2)

def generate_enhanced_grid_environment(
    n, m,
    r1_states, r2_states, r3_states, r4_states, r5_states,
    r1, r2, r3, r4, r5,
    r1_probabilities, r2_probabilities, r3_probabilities, r4_probabilities, r5_probabilities,
    r1p, r2p, r3p, r4p, r5p,
    trajectory,
    second_trajectory=None,
    cell_text=None,
    interval=200,
    h1=3,
    h2=3
):
    letter_cell_colors = {
        'W': (0, 0, 1),
        'S': (0, 1, 0),
        'V': (1, 0, 0),
        'G': (1, 1, 0),
        'Agent_2': (1, 0.5, 0)
    }

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    for i in range(m + 1):
        ax.plot([i, i], [0, n], color='black', linewidth=0.5, zorder=10)
    for j in range(n + 1):
        ax.plot([0, m], [j, j], color='black', linewidth=0.5, zorder=10)

    prob_map = {}
    belief_artists = {}
    overlay_patches = {}

    # First, render the TRUE labels (r1, r2, r3, r4, r5) at zorder=1
    true_labels_dict = {}
    true_label_artists = {}
    for label, cells, probs in zip(['W', 'S', 'V', 'G', 'Agent_2'], [r1, r2, r3, r4, r5], [r1p, r2p, r3p, r4p, r5p]):
        for s in cells:
            true_labels_dict[s] = label
            x0, y0 = s % m, n - 1 - (s // m)
            rect = plt.Rectangle((x0, y0), 1, 1, color=letter_cell_colors[label] + (1.0,), zorder=1)
            ax.add_patch(rect)
            # No text annotation for true labels - only color
            true_label_artists[s] = (rect, None)

    # Then add belief states on top at zorder=2
    for states_list, probs, letter in [
        (r1_states, r1_probabilities, 'W'),
        (r2_states, r2_probabilities, 'S'),
        (r3_states, r3_probabilities, 'V'),
        (r4_states, r4_probabilities, 'G'),
        (r5_states, r5_probabilities, 'Agent_2'),
    ]:
        base_color = letter_cell_colors[letter]
        for idx, s in enumerate(states_list):
            alpha = probs[idx]
            x0, y0 = s % m, n - 1 - (s // m)
            rect = plt.Rectangle((x0, y0), 1, 1, color=base_color + (alpha,), zorder=2)
            text = ax.annotate(f'{letter}:{probs[idx]:.1f}', (x0 + 0.5, y0 + 0.2),
                               color='black', fontsize=5, ha='center', zorder=2)
            ax.add_patch(rect)
            prob_map[s] = (letter, probs[idx])
            belief_artists[s] = (rect, text)


    # Add gray overlays on top at zorder=4 (above beliefs)
    for cell in range(n * m):
        x0, y0 = cell % m, n - 1 - (cell // m)
        grey_rect = plt.Rectangle((x0, y0), 1, 1, color='grey', alpha=0.65, zorder=4)
        ax.add_patch(grey_rect)
        overlay_patches[cell] = grey_rect

    if cell_text:
        for i in range(n):
            for j in range(m):
                txt = cell_text[i][j] or ''
                ax.text(j + 0.1, n - i - 0.1, txt, ha='left', va='top', fontsize=7)

    x = [(s % m) + 0.5 for s in trajectory]
    y = [n - 1 - (s // m) + 0.65 for s in trajectory]
    line, = ax.plot([], [], 'k-', linewidth=2)
    # small red circle at start
    start_dot, = ax.plot(x[0], y[0], 'ro', markersize=6, zorder=6)


    arrow_patches = []

    if second_trajectory:
        x2 = [(s % m) + 0.5 for s in second_trajectory]
        y2 = [n - 1 - (s // m) + 0.5 for s in second_trajectory]
        line2, = ax.plot([], [], 'r-', linewidth=1)
        arrow_patches2 = []
    else:
        x2, y2, line2, arrow_patches2 = [], [], None, []

    revealed = set()
    persistent_artists = []

    def reveal_labels(robot_cell, h):
        for s in range(n * m):
            if s in revealed:
                continue

            # Compute Manhattan distance (4-directional: up, down, left, right)
            rx, ry = robot_cell % m, robot_cell // m
            sx, sy = s % m, s // m
            manhattan_dist = abs(rx - sx) + abs(ry - sy)

            if manhattan_dist <= h:
                # Remove belief state if present
                if s in belief_artists:
                    patch, text = belief_artists.pop(s)
                    patch.set_visible(False)
                    text.set_visible(False)
                    
                # Remove gray overlay
                if s in overlay_patches:
                    overlay_patches[s].remove()
                    del overlay_patches[s]
                    
                revealed.add(s)
                
                if s in prob_map:
                    del prob_map[s]

    def init():
        line.set_data([], [])
        if line2:
            line2.set_data([], [])
        return [line, start_dot, line2] if line2 else [line, start_dot]


    def update(frame):
        if frame < len(trajectory):
            current_cell = trajectory[frame]
            reveal_labels(current_cell, h1)
            line.set_data(x[:frame + 1], y[:frame + 1])
            if frame > 0:
                dx = x[frame] - x[frame - 1]
                dy = y[frame] - y[frame - 1]
                art = ax.arrow(x[frame - 1], y[frame - 1], dx, dy, head_width=0.3, head_length=0.35, fc='k', ec='k', zorder=5)
                arrow_patches.append(art)

        if second_trajectory and frame < len(second_trajectory):
            current_cell2 = second_trajectory[frame]
            reveal_labels(current_cell2, h2)
            line2.set_data(x2[:frame + 1], y2[:frame + 1])
            if frame > 0:
                dx = x2[frame] - x2[frame - 1]
                dy = y2[frame] - y2[frame - 1]
                art2 = ax.arrow(x2[frame - 1], y2[frame - 1], dx, dy, head_width=0.3, head_length=0.35, fc='r', ec='r', zorder=5)
                arrow_patches2.append(art2)

        visible_beliefs = [artist for pair in belief_artists.values()
                           for artist in pair if artist.get_visible()]
        visible_true_labels = [artist for pair in true_label_artists.values()
                              for artist in pair if artist and artist.get_visible()]
        return [line, start_dot] + arrow_patches + \
            ([line2] + arrow_patches2 if line2 else []) + \
            visible_beliefs + visible_true_labels + persistent_artists

    ani = FuncAnimation(
        fig, update,
        frames=max(len(trajectory), len(second_trajectory) if second_trajectory else 0),
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False
    )
    
    ani.save("explore.mp4", writer="ffmpeg", fps=5, dpi=300)
    plt.close(fig)
    return ani


# Standalone usage: reads map from config.py and visualises a sample trajectory.
if __name__ == '__main__':
    from config import N, M, H, NODE_LABELS, PROP_COLOR_ROLE

    role_to_cells = {role: [] for role in ('r1', 'r2', 'r3', 'r4', 'r5')}
    for cell, props in NODE_LABELS.items():
        for prop in props:
            role = PROP_COLOR_ROLE.get(prop)
            if role:
                role_to_cells[role].append(cell)

    trajectory = [
        0, 1, 21, 41, 42, 43, 44, 45, 46, 47, 48, 49, 69, 89, 109, 129, 149,
        169, 189, 209, 229, 249, 269, 289, 309, 310, 311, 312, 292, 293, 294,
        295, 296, 297, 277, 257, 237, 217, 197, 177, 157, 137, 117, 97, 77,
        57, 37, 36, 35, 34, 33, 32, 31, 51, 52, 53,
    ]

    generate_enhanced_grid_environment(
        N, M,
        [], [], [], [], [],
        role_to_cells['r1'], role_to_cells['r2'],
        role_to_cells['r3'], role_to_cells['r4'],
        role_to_cells['r5'],
        [], [], [], [], [],
        [1], [1], [1], [1], [1],
        trajectory,
        second_trajectory=None,
        cell_text=None,
        interval=300,
        h1=H,
    )





