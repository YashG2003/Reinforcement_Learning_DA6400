import gymnasium as gym
import numpy as np
import heapq

# Constants for options
PICKUP_OPTION, DROPOFF_OPTION = 0, 1
OPTION_NAMES = ['Pickup', 'Dropoff']

# Locations of R, G, Y, B
LOCATIONS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B

def decode_state(env, state):
    return env.unwrapped.decode(state)

def get_wall_map(env):
    """Extract wall information from the environment"""
    desc = env.unwrapped.desc.astype('c').tolist()
    wall_map = np.zeros((5, 5, 4))  # 5x5 grid, 4 directions (S, N, E, W)

    for row in range(5):
        for col in range(5):
            # South (down)
            if row < 4:
                wall_map[row, col, 0] = 1
            # North (up)
            if row > 0:
                wall_map[row, col, 1] = 1
            # East (right)
            if col < 4 and desc[1 + row][2 * col + 2] == b':':
                wall_map[row, col, 2] = 1
            # West (left)
            if col > 0 and desc[1 + row][2 * col] == b':':
                wall_map[row, col, 3] = 1
    return wall_map

def dijkstra(wall_map, start, goal):
    rows, cols = 5, 5
    distances = {(r, c): float('inf') for r in range(rows) for c in range(cols)}
    distances[start] = 0
    prev = {(r, c): None for r in range(rows) for c in range(cols)}
    pq = [(0, start)]
    visited = set()

    while pq:
        dist, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            break

        r, c = current
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # S, N, E, W
        actions = [0, 1, 2, 3]

        for i, (dr, dc) in enumerate(directions):
            if wall_map[r, c, i] == 1:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    alt = dist + 1
                    if alt < distances[(nr, nc)]:
                        distances[(nr, nc)] = alt
                        prev[(nr, nc)] = (current, actions[i])
                        heapq.heappush(pq, (alt, (nr, nc)))

    # Reconstruct path with actions
    path = []
    node = goal
    while node != start and prev[node] is not None:
        prev_node, action = prev[node]
        path.append((node, action))
        node = prev_node

    path.reverse()
    return path

def precompute_paths(env):
    wall_map = get_wall_map(env)
    paths = {}

    for dest_idx, dest_pos in enumerate(LOCATIONS):
        paths[dest_idx] = {}
        for r in range(5):
            for c in range(5):
                start = (r, c)
                if start != dest_pos:
                    path = dijkstra(wall_map, start, dest_pos)
                    paths[dest_idx][start] = path

    return paths

_paths = None

def initialize_paths(env):
    global _paths
    if _paths is None:
        _paths = precompute_paths(env)
    return _paths

def option_policy(env, option_idx, state):
    taxi_row, taxi_col, pass_loc, dest_idx = decode_state(env, state)
    current_pos = (taxi_row, taxi_col)
    paths = initialize_paths(env)

    # Pickup Option
    if option_idx == PICKUP_OPTION:
        if pass_loc == 4:
            return None  # Already picked up
        target_pos = LOCATIONS[pass_loc]
        if current_pos == target_pos:
            return 4  # Pickup
        path = paths[pass_loc].get(current_pos, [])
        if path:
            _, action = path[0]
            return action
        # Fallback greedy move
        if taxi_row < target_pos[0]: return 0  # South
        if taxi_row > target_pos[0]: return 1  # North
        if taxi_col < target_pos[1]: return 2  # East
        if taxi_col > target_pos[1]: return 3  # West

    # Dropoff Option
    elif option_idx == DROPOFF_OPTION:
        if pass_loc != 4:
            return None  # No passenger in taxi
        target_pos = LOCATIONS[dest_idx]
        if current_pos == target_pos:
            return 5  # Dropoff
        path = paths[dest_idx].get(current_pos, [])
        if path:
            _, action = path[0]
            return action
        # Fallback greedy move
        if taxi_row < target_pos[0]: return 0  # South
        if taxi_row > target_pos[0]: return 1  # North
        if taxi_col < target_pos[1]: return 2  # East
        if taxi_col > target_pos[1]: return 3  # West

    return None

def option_terminates(env, option_idx, state):
    """Check if option terminates in state"""
    taxi_row, taxi_col, pass_loc, dest_idx = decode_state(env, state)
    
    # Pickup option terminates if passenger already in taxi or at passenger location
    if option_idx == PICKUP_OPTION:
        return pass_loc == 4 #or (taxi_row, taxi_col) == LOCATIONS[pass_loc]
    
    # Dropoff option terminates if passenger not in taxi or at destination
    if option_idx == DROPOFF_OPTION:
        return pass_loc != 4 #or (taxi_row, taxi_col) == LOCATIONS[dest_idx]
    
    return False


def get_consistent_options(env, state, action):
    """
    Get all options that would have selected the same action in the given state
    """
    consistent_options = []
    
    for option_idx in range(len(OPTION_NAMES)):
        option_action = option_policy(env, option_idx, state)
        if option_action == action:
            consistent_options.append(option_idx)
            
    return consistent_options
