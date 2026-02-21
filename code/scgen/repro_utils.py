import os
import random

import numpy as np


def seed_everything_from_env():
    seed = os.environ.get("SCGEN_PROCESS_SEED") or os.environ.get("SCGEN_SEED") or "1248"
    try:
        seed_value = int(seed)
    except ValueError:
        seed_value = 1248
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import tensorflow as tf

        if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
            tf.random.set_seed(seed_value)
        elif hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
            tf.compat.v1.set_random_seed(seed_value)
    except Exception:
        pass
    return seed_value
