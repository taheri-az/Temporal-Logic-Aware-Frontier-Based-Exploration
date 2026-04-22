import os
import subprocess
from Helper import *
from config import (
    N, M, H,
    ALPHA1, ALPHA2, ALPHA3,
    FORMULA_STR, ACCEPTING_STATES, TRASH_STATE,
    INITIAL_PHYSICAL_STATE, NODE_LABELS, PROP_COLOR_ROLE,
)
from explore_visual import generate_enhanced_grid_environment

# ── Grid / graph ──────────────────────────────────────────────────────────────
n, m, h                = N, M, H
alpha1, alpha2, alpha3 = ALPHA1, ALPHA2, ALPHA3
trash_state            = TRASH_STATE

grid                     = create_grid(n, m)
nodes, edges, adj_matrix = create_graph(n, m)
X_size                   = n * m

# ── DFA from LTL formula ──────────────────────────────────────────────────────
atomic_props = extract_atomic_props(FORMULA_STR)
dfa_transitions, initial_state, trash_states_set = \
    extract_dfa_transitions_with_trash_expanded(FORMULA_STR)
pruned_dfa_transitions = prune_dfa_transitions_single_ap_only(
    dfa_transitions, atomic_props
)
accepting_states = ACCEPTING_STATES

dfa_states = set()
for s1, _, s2 in dfa_transitions:
    dfa_states.update([s1, s2])

# ── Label maps ────────────────────────────────────────────────────────────────
node_labels_t = {k: set(v) for k, v in NODE_LABELS.items()}
for node in nodes.flatten():
    node = int(node)
    if node not in node_labels_t:
        node_labels_t[node] = set()

node_labels = {int(node): set() for node in nodes.flatten()}

# ── Initial product automaton (all cells unknown) ─────────────────────────────
product_graph, transitions, product_nodes, PR_adj_matrix = \
    generate_product_automaton22(nodes, edges, dfa_states, dfa_transitions, node_labels)

# ── DFA distances & commit states ─────────────────────────────────────────────
dfa_distances = compute_dfa_distances_to_accepting(
    dfa_states, pruned_dfa_transitions, accepting_states
)
print("DFA distances to accepting state:")
for state in dfa_states:
    print(f"  State {state}: {dfa_distances.get(state, float('inf'))}")

commit_states = compute_commit_states(FORMULA_STR, dot_file="large_product", fmt="pdf")
commit_states = [s for s in commit_states if s not in trash_states_set]
commit_states = [str(s) for s in commit_states]
print("commit_states", commit_states)

# ── Exploration loop ──────────────────────────────────────────────────────────
current_physical_state = INITIAL_PHYSICAL_STATE
current_dfa_state      = initial_state
visited            = set()
full_traj          = []
full_physical_traj = [current_physical_state]

while True:
    print(f"\nCurrent state: ({current_physical_state}, {current_dfa_state})")

    # Step 1: reveal cells within sensor range
    h_neighbors = get_states_within_h_distance(m, n, current_physical_state, h)
    new_nodes, new_edges = find_new_physical_nodes_edges(
        visited, product_nodes, adj_matrix, product_graph
    )
    for node in h_neighbors:
        node_labels[node] = node_labels_t.get(node, set())
        visited.add(node)

    # Step 2: rebuild product automaton with updated labels
    product_graph, transitions, product_nodes, PR_adj_matrix = \
        generate_product_automaton22(nodes, edges, dfa_states, dfa_transitions, node_labels)

    # Step 3: check if accepting path is already reachable
    current_product_state = (current_physical_state, current_dfa_state)
    accepting_path = find_shortest_path_to_accepting(
        current_product_state, {'accept_all'}, transitions
    )
    if accepting_path:
        print("Accepting path found! Executing...")
        for state in accepting_path[1:]:
            current_physical_state, current_dfa_state = state
            full_traj.append(state)
            full_physical_traj.append(int(current_physical_state))
            print(" →", state)
        break

    # Step 4: detect frontiers
    unknown   = set(range(n * m)) - visited
    frontiers = detect_frontiers_e(n, m, visited, unknown)
    if not frontiers:
        print("No frontiers left, and no satisfying path found.")
        break

    # Step 5: score frontiers
    I_x_dict = {
        x: len(set(get_states_within_h_distance(m, n, x, h)) - visited)
        for x in frontiers
    }
    weights, best_paths = {}, {}
    for x in frontiers:
        w, sp = compute_frontier_commit(
            x=x,
            product_graph=product_graph,
            start_cell=current_physical_state,
            start_dfa_state=current_dfa_state,
            accepting_states=accepting_states,
            commit_states=commit_states,
            trash_state=trash_state,
            delta_phi=delta_phi,
            I_x=I_x_dict[x],
            X_size=X_size,
            dfa_distance=dfa_distances,
            alpha1=alpha1,
            alpha2=alpha2,
            alpha3=alpha3,
        )
        weights[x]    = w
        best_paths[x] = sp

    valid_frontiers = [x for x in frontiers if best_paths[x] is not None]
    if not valid_frontiers:
        print("No reachable frontiers left, stopping exploration.")
        break

    best_frontier = max(valid_frontiers, key=lambda x: weights[x])
    path = [s for (s, q) in best_paths[best_frontier]]

    if current_dfa_state == 'accepting_all':
        break

    # Step 6: execute path and update DFA state
    for step in path[1:]:
        label = node_labels.get(step, set())
        current_dfa_state      = get_next_dfa_state(current_dfa_state, label, dfa_transitions)
        current_physical_state = step
        for node in get_states_within_h_distance(m, n, current_physical_state, h):
            node_labels[node] = node_labels_t.get(node, set())
            visited.add(node)
        full_traj.append((step, current_dfa_state))
        full_physical_traj.append(step)

print("\nFinal physical trajectory:")
print(full_physical_traj)

# ── Visualization ─────────────────────────────────────────────────────────────
role_to_cells = {role: [] for role in ('r1', 'r2', 'r3', 'r4', 'r5')}
for cell, props in NODE_LABELS.items():
    for prop in props:
        role = PROP_COLOR_ROLE.get(prop)
        if role:
            role_to_cells[role].append(cell)

generate_enhanced_grid_environment(
    n, m,
    [], [], [], [], [],                              # no belief states
    role_to_cells['r1'], role_to_cells['r2'],
    role_to_cells['r3'], role_to_cells['r4'],
    role_to_cells['r5'],
    [], [], [], [], [],                              # no belief probabilities
    [1], [1], [1], [1], [1],                         # true-label placeholders
    full_physical_traj,
    second_trajectory=None,
    cell_text=None,
    interval=200,
    h1=h,
)

video_path = os.path.abspath("explore.mp4")
print(f"\nPlaying {video_path}")
subprocess.Popen(["xdg-open", video_path])
