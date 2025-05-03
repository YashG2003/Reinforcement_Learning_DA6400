import gymnasium as gym
import numpy as np
import heapq

NAV_R, NAV_G, NAV_Y, NAV_B, PICKUP, DROPOFF = 0, 1, 2, 3, 4, 5
OPTION_NAMES = ['Nav-R', 'Nav-G', 'Nav-Y', 'Nav-B', 'Pickup', 'Dropoff']

OPTION_TARGETS = {
    NAV_R: 0,  # R(ed)
    NAV_G: 1,  # G(reen)
    NAV_Y: 2,  # Y(ellow)
    NAV_B: 3,  # B(lue)
}

# Locations of R, G, Y, B
LOCATIONS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B

def decode_state(env, state):
    return env.unwrapped.decode(state)

def get_wall_map(env):
    """Extract wall information from the environment"""
    desc = env.unwrapped.desc.astype('c').tolist()
    wall_map = np.zeros((5, 5, 4))  # 5x5 grid, 4 directions (S, N, E, W)
    
    # Check walls in all cells
    for row in range(5):
        for col in range(5):
            # Check south wall (row + 1)
            if row < 4:
                wall_map[row, col, 0] = 1  # Assume can go south
            # Check north wall (row - 1)
            if row > 0:
                wall_map[row, col, 1] = 1  # Assume can go north
            # Check east wall (col + 1)
            if col < 4:
                if desc[1 + row][2 * col + 2] == b':':
                    wall_map[row, col, 2] = 1  # Can go east
            # Check west wall (col - 1)
            if col > 0:
                if desc[1 + row][2 * col] == b':':
                    wall_map[row, col, 3] = 1  # Can go west
    
    return wall_map

def dijkstra(wall_map, start, goal):
    """Find shortest path from start to goal using Dijkstra's algorithm"""
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
        # Check neighbors (S, N, E, W) if no wall
        directions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]  # Current, S, N, E, W
        actions = [None, 0, 1, 2, 3]  # None, S, N, E, W
        
        for i, (dr, dc) in enumerate(directions[1:], 1):
            if i-1 < 4 and wall_map[r, c, i-1] == 1:  # Check if there's no wall
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
    while node != start:
        prev_node, action = prev[node]
        path.append((node, action))
        node = prev_node
    
    path.reverse()
    return path

def precompute_paths(env):
    """Precompute paths from every cell to each destination"""
    wall_map = get_wall_map(env)
    paths = {}
    
    for dest_idx, dest_pos in enumerate(LOCATIONS):
        paths[dest_idx] = {}
        for r in range(5):
            for c in range(5):
                start = (r, c)
                if start != dest_pos:  # No need for path if already at destination
                    path = dijkstra(wall_map, start, dest_pos)
                    paths[dest_idx][start] = path
    
    return paths

# Global variable to store precomputed paths
_paths = None

def initialize_paths(env):
    """Initialize paths if not already done"""
    global _paths
    if _paths is None:
        _paths = precompute_paths(env)
    return _paths

def option_policy(env, option_idx, state):
    """Return the action to take for the given option and state"""
    taxi_row, taxi_col, pass_loc, dest_idx = decode_state(env, state)
    
    # Navigation options
    if option_idx in OPTION_TARGETS:
        target = OPTION_TARGETS[option_idx]
        target_pos = LOCATIONS[target]
        
        # Terminate if at target
        if (taxi_row, taxi_col) == target_pos:
            return None
        
        # Get precomputed path
        paths = initialize_paths(env)
        current_pos = (taxi_row, taxi_col)
        
        if current_pos in paths[target]:
            path = paths[target][current_pos]
            if path:
                # Return the next action from the path
                _, action = path[0]
                return action
        
        # If no path found (shouldn't happen), use simple navigation
        target_row, target_col = target_pos
        if taxi_row < target_row:
            return 0  # South
        if taxi_row > target_row:
            return 1  # North
        if taxi_col < target_col:
            return 2  # East
        if taxi_col > target_col:
            return 3  # West
    
    # Pickup option
    if option_idx == PICKUP:
        if pass_loc == 4:  # Already in taxi
            return None
        if (taxi_row, taxi_col) == LOCATIONS[pass_loc]:
            return 4  # Pickup
        return None  # Don't allow illegal pickup
    
    # Dropoff option
    if option_idx == DROPOFF:
        if pass_loc != 4:  # Not in taxi
            return None
        if (taxi_row, taxi_col) == LOCATIONS[dest_idx]:
            return 5  # Dropoff
        return None  # Don't allow illegal dropoff
    
    return None

def option_terminates(env, option_idx, state):
    """Check if option terminates in state"""
    taxi_row, taxi_col, pass_loc, dest_idx = decode_state(env, state)
    
    # Navigation options terminate at target
    if option_idx in OPTION_TARGETS:
        target = OPTION_TARGETS[option_idx]
        target_pos = LOCATIONS[target]
        return (taxi_row, taxi_col) == target_pos
    
    # Pickup option terminates if passenger already in taxi or at passenger location
    if option_idx == PICKUP:
        return pass_loc == 4 or (taxi_row, taxi_col) == LOCATIONS[pass_loc]
    
    # Dropoff option terminates if passenger not in taxi or at destination
    if option_idx == DROPOFF:
        return pass_loc != 4 or (taxi_row, taxi_col) == LOCATIONS[dest_idx]
    
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

