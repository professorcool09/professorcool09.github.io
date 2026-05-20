"""
adapter.py
==========
Bridges a 2-player Catan game with a model trained on 4-player games.

The Problem
-----------
The new model (FFF38MR2) was trained on 4-player games, so it expects a
1002-feature observation vector that includes road/settlement/city/resource
slots for players P2 and P3.

Our website runs a 2-player game (BLUE vs RED = P0 and P1), which only
produces a 614-feature observation vector.

The Solution
------------
Build the full 1002-feature observation from the 2-player game by looking
up every feature name in the 4-player ordering:
  - If the feature exists in the 2-player game's `create_sample()` dict,
    use that value.
  - If it doesn't (i.e. it's a P2 or P3 slot), use 0.0 — because those
    players don't exist, they have no roads, no settlements, no resources.

This is mathematically correct: the model learned that P2/P3 features = 0
simply means those players have nothing.  In a 2-player game that is always
true, so the model sees exactly what it would see for "ghost" players.

Usage in app.py
---------------
    from adapter import build_4p_obs, FEATURES_4P

    # Replace the old FEATURES line with:
    #   FEATURES = get_feature_ordering(num_players=2)   <- REMOVE
    # The adapter handles feature ordering internally.

    # In bot_decide(), replace:
    #   obs_raw = np.array([float(sample[f]) for f in FEATURES], ...)
    # with:
    #   obs_raw = build_4p_obs(sample)
"""

import numpy as np
from catanatron_gym.features import get_feature_ordering

# Build the feature ordering once at import time — these are module-level
# constants so they are only computed once no matter how many games are played.

# All 1002 feature names in the order the 4-player model expects
FEATURES_4P = get_feature_ordering(num_players=4)

# All 614 feature names for a 2-player game (used to know which names exist)
_FEATURES_2P_SET = set(get_feature_ordering(num_players=2))

# Pre-compute a boolean mask: True for features that exist in 2-player games,
# False for the 388 P2/P3 slots that should always be 0.
_FEATURE_EXISTS = np.array(
    [f in _FEATURES_2P_SET for f in FEATURES_4P], dtype=bool
)

# Pre-compute the indices of features that need to be filled from the sample
_FILL_INDICES = [i for i, f in enumerate(FEATURES_4P) if f in _FEATURES_2P_SET]
_FILL_NAMES   = [f for f in FEATURES_4P if f in _FEATURES_2P_SET]


def build_4p_obs(sample: dict) -> np.ndarray:
    """
    Build a 1002-element float32 observation vector suitable for the
    4-player model, using data from a 2-player game's create_sample() dict.

    Parameters
    ----------
    sample : dict
        The dict returned by catanatron_gym's create_sample(game, color).
        Contains 614 named features for the 2-player game.

    Returns
    -------
    np.ndarray, shape (1002,), dtype float32
        Full 4-player observation with P2/P3 features set to 0.0.
    """
    obs = np.zeros(len(FEATURES_4P), dtype=np.float32)
    for i, name in zip(_FILL_INDICES, _FILL_NAMES):
        obs[i] = float(sample.get(name, 0.0))
    return obs


def obs_info() -> dict:
    """Return a summary dict useful for debugging."""
    return {
        "total_features_4p":    len(FEATURES_4P),
        "shared_features_2p":   len(_FILL_INDICES),
        "zeroed_p2_p3_features": len(FEATURES_4P) - len(_FILL_INDICES),
    }
