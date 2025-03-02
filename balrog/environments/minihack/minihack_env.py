from typing import Optional

import gym
import minihack  # NOQA: F401

from balrog.environments.nle import NLELanguageWrapper
from balrog.environments.wrappers import GymV21CompatibilityV0, NLETimeLimit

MINIHACK_ENVS = []
for env_spec in gym.envs.registry.all():
    id = env_spec.id
    if id.split("-")[0] == "MiniHack":
        MINIHACK_ENVS.append(id)


def make_minihack_env(env_name, task, config, render_mode: Optional[str] = None):
    import logging
    logging.info("Initializing MiniHack environment: env_name=%s, task=%s", env_name, task)
    minihack_kwargs = dict(config.envs.minihack_kwargs)
    skip_more = minihack_kwargs.pop("skip_more", False)
    logging.info("Using minihack_kwargs: %s with skip_more=%s", minihack_kwargs, skip_more)
    vlm = True if config.agent.max_image_history > 0 else False
    logging.info("Visual logging mode set to: %s", vlm)

    env = gym.make(
        task,
        observation_keys=[
            "glyphs",
            "blstats",
            "tty_chars",
            "inv_letters",
            "inv_strs",
            "tty_cursor",
            "tty_colors",
        ],
        **minihack_kwargs,
    )
    logging.info("Gym environment created for task: %s", task)

    env = NLELanguageWrapper(env, vlm=vlm, skip_more=skip_more)
    logging.info("Applied NLELanguageWrapper to the environment")

    # wrap NLE with timeout
    env = NLETimeLimit(env)
    logging.info("Applied NLETimeLimit wrapper to the environment")

    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)
    logging.info("Applied GymV21CompatibilityV0 wrapper with render_mode: %s", render_mode)

    logging.info("MiniHack environment creation complete for env_name=%s, task=%s", env_name, task)

    # raise Exception("Stop here")
    return env
