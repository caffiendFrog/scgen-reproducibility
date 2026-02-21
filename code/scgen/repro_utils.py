import os
import random

import numpy as np


DEFAULT_SEED = 4039


def resolve_seed(seed_value, default=DEFAULT_SEED):
    try:
        return int(seed_value)
    except (TypeError, ValueError):
        return default


def get_seed_from_env(default=DEFAULT_SEED):
    return resolve_seed(os.environ.get("SCGEN_PROCESS_SEED") or os.environ.get("SCGEN_SEED"), default=default)


def apply_reproducibility_env(env, seed):
    seed_text = str(resolve_seed(seed))
    env["SCGEN_PROCESS_SEED"] = seed_text
    env["PYTHONHASHSEED"] = seed_text
    env["NUMPY_SEED"] = seed_text
    env["TF_SEED"] = seed_text
    if env.get("SCGEN_ENABLE_DETERMINISM", "1") != "0":
        env.setdefault("TF_DETERMINISTIC_OPS", "1")
        env.setdefault("TF_CUDNN_DETERMINISTIC", "1")


def set_tf_graph_seed_from_env(tf_module):
    seed_value = get_seed_from_env()
    if hasattr(tf_module, "compat") and hasattr(tf_module.compat, "v1"):
        tf_module.compat.v1.set_random_seed(seed_value)
    return seed_value


def get_tf_op_seed():
    return get_seed_from_env()


def seed_everything_from_env():
    seed_value = get_seed_from_env()
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import tensorflow as tf

        if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
            tf.random.set_seed(seed_value)
        set_tf_graph_seed_from_env(tf)
    except Exception:
        pass
    return seed_value
