import spot
import buddy
from graphviz import Digraph
from itertools import product
import re
import networkx as nx
import numpy as np
from collections import deque
import math

def create_grid(n, m):
    grid = np.zeros((n, m), dtype=int)
    return grid

# Function to create the graph representation of the grid
def create_graph(n, m):
    num_nodes = n * m
    nodes = np.arange(num_nodes).reshape(n, m)
    edges = []

    # Add horizontal edges
    for i in range(n):
        for j in range(m-1):
            edges.append((nodes[i][j], nodes[i][j+1]))
            edges.append((nodes[i][j+1], nodes[i][j]))

    # Add vertical edges
    for j in range(m):
        for i in range(n-1):
            edges.append((nodes[i][j], nodes[i+1][j]))
            edges.append((nodes[i+1][j], nodes[i][j]))

    # Add stay action
    for i in range(n):
        for j in range(m):
            edges.append((nodes[i][j], nodes[i][j]))

    # Create the adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges:
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1

    return nodes, edges, adj_matrix



def get_states_within_h_distance(m, n, current_state, h):
    # Function to convert state number to row and column
    def state_to_row_col(state):
        return divmod(state, n)

    # Function to convert row and column to state number
    def row_col_to_state(row, col):
        if 0 <= row < m and 0 <= col < n:
            return row * n + col
        return None

    # Function to get all adjacent states of a given state
    def get_adjacent_states(state):
        row, col = state_to_row_col(state)
        adjacent_states = []
        for r, c in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            adjacent_state = row_col_to_state(r, c)
            if adjacent_state is not None:
                adjacent_states.append(adjacent_state)
        return adjacent_states

    # BFS to find all states within h distance
    visited = set()
    queue = [(current_state, 0)]  # Each element is a tuple (state, distance)
    while queue:
        state, distance = queue.pop(0)
        if distance > h:
            break
        visited.add(state)
        for next_state in get_adjacent_states(state):
            if next_state not in visited:
                queue.append((next_state, distance + 1))

    return list(visited)

# Function to extract observations from DFA content
def extract_observations(dfa_content):
    observations = set(re.findall(r'\((.*?)\)', dfa_content))
    return observations

def compute_commit_states(phi, dot_file=None, fmt="pdf"):
    """
    Given an LTL formula phi, translate it to a deterministic, complete DFA,
    build its full self-product automaton,
    and compute the commit states.

    Commit states q ≠ init_state such that from (init_state, q) there is a path
    to (p_accept, q_non_accept) in the product.

    Args:
      phi (str): LTL formula string.
      dot_file (str or None): If provided, save the product graph visualization to this file.
      fmt (str): Format for the graph output, default 'pdf'.

    Returns:
      List[int]: The list of commit states in the original DFA.
    """

    # 1) Translate formula → deterministic, complete DFA
    dfa = spot.translate(phi, 'deterministic', 'complete')
    n = dfa.num_states()
    bdd_dict = dfa.get_dict()
    init_state = dfa.get_init_state_number()

    # 2) Identify accepting and non-accepting states
    accept_states = {s for s in range(n) if dfa.state_is_accepting(s)}
    non_accept_states = set(range(n)) - accept_states

    # 3) Prepare false BDD constant
    false_bdd = buddy.bddfalse

    # 4) Initialize graphviz Digraph if requested
    dot = None
    if dot_file:
        dot = Digraph(comment=f"Full Self-Product of '{phi}'",
                      filename=dot_file,
                      format=fmt)
        for p in range(n):
            for q in range(n):
                dot.node(f"{p},{q}", label=f"{p},{q}", shape="circle")

    # 5) Build adjacency for the product automaton
    adj = {(p, q): [] for p in range(n) for q in range(n)}
    for p in range(n):
        for tr_p in dfa.out(p):
            for q in range(n):
                for tr_q in dfa.out(q):
                    joint = tr_p.cond & tr_q.cond
                    if joint == false_bdd:
                        continue
                    p2, q2 = tr_p.dst, tr_q.dst
                    adj[(p, q)].append((p2, q2))
                    if dot:
                        lbl = spot.bdd_format_formula(bdd_dict, joint)
                        dot.edge(f"{p},{q}", f"{p2},{q2}", label=lbl)

    # 6) Compute commit states
    commit_states = []
    for q in range(n):
        if q == init_state:
            continue
        start = (init_state, q)
        seen = {start}
        stack = [start]
        found = False
        while stack and not found:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
                    p2, q2 = v
                    if (p2 in accept_states) and (q2 in non_accept_states):
                        found = True
                        break
        if found:
            commit_states.append(q)

 
    return commit_states






def generate_product_automaton22(nodes, edges, dfa_states, dfa_transitions,
                                 node_labels, atomic_props):
    """
    Build the product automaton (grid x DFA).

    Observation strings are generated using `atomic_props`, which must be the
    DFA's own AP ordering, so that every generated string matches a DFA edge.
    """

    # --- Build DFA dictionary for fast lookup ---
    dfa_dict = {}
    for q, obs_list, q_next in dfa_transitions:
        for formula in obs_list:  # allow multiple formulas
            dfa_dict[(q, formula)] = q_next

    # --- Build adjacency dict from edge list ---
    adj = {int(v): [] for v in nodes.flatten()}
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))  # if undirected

    # --- Enumerate all product nodes ---
    product_nodes = [(int(v), q) for v in nodes.flatten() for q in dfa_states]

    # --- Build product transitions ---
    transitions = {pn: set() for pn in product_nodes}  # use set to remove duplicates

    for v in nodes.flatten():
        v = int(v)
        if v not in adj:
            continue
        for v_next in adj[v]:
            obs = _obs_formula(node_labels.get(v_next, set()), atomic_props)
            for q in dfa_states:
                if (q, obs) in dfa_dict:
                    transitions[(v, q)].add((v_next, dfa_dict[(q, obs)]))

    # Convert sets to lists for output
    for k in transitions:
        transitions[k] = list(transitions[k])

    # --- Build NetworkX graph ---
    product_graph = nx.DiGraph()
    product_graph.add_nodes_from(product_nodes)
    for src, dst_list in transitions.items():
        for dst in dst_list:
            product_graph.add_edge(src, dst)

    return product_graph, transitions, product_nodes, None


def prune_dfa_transitions_single_ap_only(dfa_transitions, atomic_props):
    """
    Filters DFA transitions to keep only those where exactly one atomic proposition is True.

    Args:
        dfa_transitions (list of tuples): Each tuple is (current_state, [condition], next_state)
        atomic_props (list of str): List of atomic propositions, e.g., ['s', 'p', 'd']

    Returns:
        list of tuples: Filtered DFA transitions
    """
    pruned_transitions = []

    for (current_state, conditions, next_state) in dfa_transitions:
        for cond in conditions:
            # Count number of atomic props that are True in this condition
            true_count = 0
            for ap in atomic_props:
                # Check if atomic proposition appears positively (without !)
                if f'{ap}' in cond and f'!{ap}' not in cond:
                    true_count += 1
            if true_count == 1:
                pruned_transitions.append((current_state, [cond], next_state))

    return pruned_transitions



def compute_dfa_distances_to_accepting(dfa_states, dfa_transitions, accepting_states):
    """
    Compute the shortest distance (in number of transitions) from each DFA state
    to the nearest accepting state, using a list of transitions.

    Parameters
    ----------
    dfa_states : list
        List of DFA states.
    dfa_transitions : list
        List of tuples (q, [formula], q_next)
    accepting_states : set or list
        Accepting DFA states.

    Returns
    -------
    dfa_distances : dict
        Mapping: state -> distance to nearest accepting state (int or float('inf') if unreachable)
    """

    # Initialize distances: all infinity
    dfa_distances = {q: float('inf') for q in dfa_states}

    # BFS queue starting from all accepting states
    queue = deque()
    for acc in accepting_states:
        if acc in dfa_states:
            dfa_distances[acc] = 0
            queue.append(acc)

    # Build a reverse adjacency map: next_state -> list of previous states
    reverse_adj = {q: [] for q in dfa_states}
    for q, obs_list, q_next in dfa_transitions:
        reverse_adj[q_next].append(q)

    # BFS over the reverse DFA graph
    while queue:
        current = queue.popleft()
        current_dist = dfa_distances[current]

        for prev in reverse_adj[current]:
            if dfa_distances[prev] == float('inf'):
                dfa_distances[prev] = current_dist + 1
                queue.append(prev)

    # print("dfaaaa",dfa_distances)
    return dfa_distances



def get_states_within_h_distance_with_diagonals(m, n, current_physical_state, h):
    """
    Get all grid states within Chebyshev distance h (diagonals allowed).

    Parameters
    ----------
    m, n : int
        Grid size (m rows, n columns)
    current_physical_state : int or tuple
        Either linear index or (row, col)
    h : int
        Distance threshold

    Returns
    -------
    neighbors : set
        Set of physical states (same format as input)
    """

    # Convert to (row, col) if needed
    if isinstance(current_physical_state, int):
        r0 = current_physical_state // n
        c0 = current_physical_state % n
        return_as_index = True
    else:
        r0, c0 = current_physical_state
        return_as_index = False

    neighbors = set()

    for r in range(max(0, r0 - h), min(m, r0 + h + 1)):
        for c in range(max(0, c0 - h), min(n, c0 + h + 1)):
            # Chebyshev distance
            if max(abs(r - r0), abs(c - c0)) <= h:
                if return_as_index:
                    neighbors.add(r * n + c)
                else:
                    neighbors.add((r, c))

    return neighbors



def find_new_physical_nodes_edges(visited, product_nodes, adj_matrix, product_graph):
    """
    Find newly discovered product nodes and edges from unvisited states.

    Parameters
    ----------
    visited : set
        Visited product states (v, q)
    product_nodes : list
        All product states
    adj_matrix : np.ndarray
        Product adjacency matrix
    product_graph : nx.DiGraph
        Product graph

    Returns
    -------
    new_nodes : set
        Newly discovered product states
    new_edges : set
        Newly discovered edges (u, v)
    """

    new_nodes = set()
    new_edges = set()

    node_to_idx = {node: i for i, node in enumerate(product_nodes)}

    for u in visited:
        if u not in node_to_idx:
            continue

        i = node_to_idx[u]
        successors = product_graph.successors(u)

        for v in successors:
            if v not in visited:
                new_nodes.add(v)
                new_edges.add((u, v))

    return new_nodes, new_edges




def find_shortest_path_to_accepting(current_product_state, accepting_dfa_states, transitions):
    """
    Find the shortest path from current product state to any accepting DFA state.

    Parameters
    ----------
    current_product_state : tuple
        (physical_state, dfa_state)
    accepting_dfa_states : set
        Accepting DFA states (e.g., {'accept_all'})
    transitions : dict
        Product transitions: (v, q) -> list of (v', q')

    Returns
    -------
    path : list or None
        List of product states forming the path, or None if unreachable
    """

    queue = deque()
    queue.append(current_product_state)

    parent = {current_product_state: None}
    visited = set([current_product_state])

    while queue:
        current = queue.popleft()
        _, q = current

        # Check acceptance
        if q in accepting_dfa_states:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]

        for nxt in transitions.get(current, []):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = current
                queue.append(nxt)

    return None



def detect_frontiers_e(n, m, visited, unknown):
    """
    Frontier cells are VISITED cells that are 4-connected
    to at least one UNKNOWN cell.
    """

    frontiers = set()

    for v in visited:
        r = v // m
        c = v % m

        neighbors = [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1)
        ]

        for rr, cc in neighbors:
            if 0 <= rr < n and 0 <= cc < m:
                u = rr * m + cc
                if u in unknown:
                    frontiers.add(v)   # <-- ADD THE VISITED CELL
                    break              # no need to check other neighbors

    return frontiers



def normalize_dfa_transitions(dfa_transitions):
    """
    Ensures DFA states are integers everywhere.
    """
    normalized = {}
    for s, trans in dfa_transitions.items():
        s = int(s)
        normalized[s] = {}
        for ap, nxt in trans.items():
            normalized[s][ap] = int(nxt)
    return normalized



def build_product_graph(n, m, dfa_transitions, node_labels):
    """
    Builds the product automaton graph safely.
    """
    G = nx.DiGraph()

    grid_states = range(n * m)

    for x in grid_states:
        aps = node_labels[x]

        for q, trans in dfa_transitions.items():
            q = int(q)
            s = (x, q)
            G.add_node(s)

            for ap in aps:
                if ap in trans:
                    q_next = trans[ap]
                    s_next = (x, q_next)
                    G.add_edge(s, s_next, weight=1)

    return G




def compute_frontier_value(
    x, product_graph, start_cell, start_dfa_state, accepting_states,
    commit_states, trash_state, delta_phi, I_x, X_size, dfa_distance,
    alpha1, alpha2, alpha3
):
    """
    Compute frontier value V(x) and the trajectory sp that achieves it.

    A frontier cell x appears in the product as one copy (x, q) per DFA state.
    Every non-trash copy is scored, and the best V over all copies is returned,
    so a frontier reachable only via a longer but task-advancing detour still
    gets credit for that detour.

    Returns:
        Vx: float  (-inf if unreachable / unsafe)
        sp: list of product states (or None)
    """
    start_node = (start_cell, start_dfa_state)

    frontier_nodes = [
        (cell, q) for (cell, q) in product_graph.nodes()
        if cell == x and q != trash_state
    ]
    if not frontier_nodes:
        return float('-inf'), None

    best_Vx = float('-inf')
    best_sp = None

    for target in frontier_nodes:
        try:
            sp = nx.shortest_path(product_graph, source=start_node, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        q0 = sp[0][1]
        qf = sp[-1][1]
        if qf == trash_state:
            continue

        if qf in commit_states:
            Omega = -alpha1 * X_size / float(alpha2)
        else:
            Omega = delta_phi(q0, qf, dfa_distance)

        Wp = len(sp) - 1
        if Wp == 0:
            Wp = 1

        Vx = (alpha1 * I_x + alpha2 * Omega) / (Wp ** alpha3)
        if Vx > best_Vx:
            best_Vx = Vx
            best_sp = sp

    return best_Vx, best_sp


def delta_phi(q_start, q_final, dfa_distances):
    if q_start not in dfa_distances or q_final not in dfa_distances:
        return float('-inf')

    d0 = dfa_distances[q_start]
    df = dfa_distances[q_final]

    if d0 == float('inf') or df == float('inf'):
        return float('-inf')

    return d0 - df



def get_next_dfa_state(current_dfa_state, node_label, dfa_transitions, atomic_props):
    """
    Compute the next DFA state given the current state and node label.

    Parameters
    ----------
    current_dfa_state : str
        Current DFA state.
    node_label : set
        Set of atomic propositions active at the node, e.g., {'s', 'd'}.
    dfa_transitions : list of tuples
        DFA transitions of the form (q, [formula], q_next).
    atomic_props : list
        AP ordering used by the DFA's transition formulas.

    Returns
    -------
    next_state : str or None
        Next DFA state if a transition exists, otherwise None.
    """
    obs_formula = _obs_formula(node_label, atomic_props)

    for q, formulas, q_next in dfa_transitions:
        if q != current_dfa_state or len(formulas) != 1:
            continue
        if formulas[0] == obs_formula:
            return q_next

    return None


def extract_dfa_transitions_with_trash_expanded(formula):
    # Translate formula into deterministic complete DFA
    dfa = spot.translate(formula, 'deterministic', 'complete')
    
    # Get the BDD dictionary
    bdd_dict = dfa.get_dict()
    num_states = dfa.num_states()

    # Identify Trash states (sink non-accepting states)
    is_sink = [False] * num_states
    trash_states_set = set()  # Store original state indices that are trash
    for s in range(num_states):
        outgoing = list(dfa.out(s))
        if len(outgoing) == 1:
            tr = outgoing[0]
            if tr.dst == s and tr.cond == buddy.bddtrue and not dfa.state_is_accepting(s):
                is_sink[s] = True
                trash_states_set.add(s)

    # Map state index to names
    state_names = {}
    for i in range(num_states):
        if dfa.state_is_accepting(i):
            state_names[i] = "accept_all"
        elif is_sink[i]:
            state_names[i] = "Trash"
        else:
            state_names[i] = str(i)

    # Extract initial state index and name
    initial_state_index = dfa.get_init_state_number()
    initial_state_name = state_names[initial_state_index]
    # print(f"Initial state index: {initial_state_index}")
    # print(f"Initial state name: {initial_state_name}")

    # Get atomic propositions actually used in the automaton
    atomic_props = [str(ap) for ap in dfa.ap()]

    # Generate all possible valuations (full minterms)
    all_valuations = list(product([False, True], repeat=len(atomic_props)))

    # Helper: convert valuation to formula string
    def valuation_to_formula(valuation):
        return ' && '.join([prop if val else f'!{prop}' for prop, val in zip(atomic_props, valuation)])

    # Helper: check if valuation satisfies the transition condition
    def valuation_satisfies(cond_bdd, valuation):
        val_bdd = buddy.bddtrue
        for prop, val in zip(atomic_props, valuation):
            var_num = bdd_dict.varnum(prop)
            var_bdd = buddy.bdd_ithvar(var_num)
            if not val:
                var_bdd = buddy.bdd_not(var_bdd)
            val_bdd = buddy.bdd_and(val_bdd, var_bdd)
        product_bdd = buddy.bdd_and(cond_bdd, val_bdd)
        return product_bdd != buddy.bddfalse

    # Extract and expand transitions
    expanded_transitions = []
    for s in range(num_states):
        for tr in dfa.out(s):
            src = state_names[s]
            dst = state_names[tr.dst]
            cond_bdd = tr.cond

            for valuation in all_valuations:
                if valuation_satisfies(cond_bdd, valuation):
                    cond_str = valuation_to_formula(valuation)
                    expanded_transitions.append((src, [cond_str], dst))

    return expanded_transitions, initial_state_name, trash_states_set



def extract_atomic_props(formula):
    # Match sequences of letters (assume atomic props are alphabetic)
    # and exclude boolean operators and keywords
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)

    # Common LTL keywords and operators to exclude
    keywords = {
        'X', 'G', 'F', 'U', 'R', 'W', 'M', '&&', '||', '!', 'true', 'false',
        'True', 'False', 'not', 'and', 'or'
    }

    # Filter out keywords, keep only propositions
    atomic_props = sorted(set(tok for tok in tokens if tok not in keywords))
    return atomic_props


def extract_atomic_props_from_dfa(dfa_transitions):
    """
    Extract atomic propositions in the exact order used by the DFA's own
    transition formulas, so observation strings match DFA edge labels.
    """
    if not dfa_transitions:
        return []

    first_formula = dfa_transitions[0][1][0]
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', first_formula)
    keywords = {'X', 'G', 'F', 'U', 'R', 'W', 'M', 'true', 'false',
                'True', 'False', 'not', 'and', 'or'}

    atomic_props, seen = [], set()
    for tok in tokens:
        if tok not in keywords and tok not in seen:
            atomic_props.append(tok)
            seen.add(tok)
    return atomic_props


def _obs_formula(label_set, atomic_props):
    """
    Convert a set of active propositions into the DFA observation string,
    e.g. {'p'} with aps ['s','p','d'] -> "!s && p && !d".
    """
    if not atomic_props:
        return 'true'
    return ' && '.join(ap if ap in label_set else f"!{ap}" for ap in atomic_props)
