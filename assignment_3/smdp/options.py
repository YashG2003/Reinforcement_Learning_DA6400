import gymnasium as gym

NAV_R, NAV_G, NAV_Y, NAV_B, PICKUP, DROPOFF = 0, 1, 2, 3, 4, 5
OPTION_NAMES = ['Nav-R', 'Nav-G', 'Nav-Y', 'Nav-B', 'Pickup', 'Dropoff']

OPTION_TARGETS = {
    NAV_R: 0,  # R(ed)
    NAV_G: 1,  # G(reen)
    NAV_Y: 2,  # Y(ellow)
    NAV_B: 3,  # B(lue)
}

def decode_state(env, state):
    return env.unwrapped.decode(state)

def option_policy(env, option_idx, state):
    taxi_row, taxi_col, pass_loc, dest_idx = decode_state(env, state)
    locs = [(0,0), (0,4), (4,0), (4,3)]  # R, G, Y, B

    # Navigation options
    if option_idx in OPTION_TARGETS:
        target = OPTION_TARGETS[option_idx]
        target_row, target_col = locs[target]
        # Terminate if at target
        if (taxi_row, taxi_col) == (target_row, target_col):
            return None
        # Otherwise, move towards target
        if taxi_row < target_row:
            return 0  # South
        if taxi_row > target_row:
            return 1  # North
        if taxi_col < target_col:
            return 2  # East
        if taxi_col > target_col:
            return 3  # West

    # Pickup option: only execute if passenger not in taxi, and at passenger location
    if option_idx == PICKUP:
        # Terminate if already in taxi
        if pass_loc == 4:
            return None
        # Only execute if taxi at passenger location
        if (taxi_row, taxi_col) == locs[pass_loc]:
            return 4  # Pickup
        else:
            return None  # Don't allow illegal pickup

    # Dropoff option: only execute if passenger in taxi, and at destination
    if option_idx == DROPOFF:
        # Terminate if passenger not in taxi
        if pass_loc != 4:
            return None
        # Only execute if taxi at destination
        if (taxi_row, taxi_col) == locs[dest_idx]:
            return 5  # Dropoff
        else:
            return None  # Don't allow illegal dropoff

    return None
