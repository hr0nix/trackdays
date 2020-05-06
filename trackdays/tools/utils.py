import gym

from tf_agents.environments import suite_gym

import trackdays.envs.race_circuit


def create_race_circuit_gym_env(offscreen=True):
    env_config = {}
    if offscreen:
        env_config.update({
            'offscreen_rendering': True,
        })
    return gym.make('racecircuit-v0', config=env_config)


def create_race_circuit_tf_agents_env(offscreen=True):
    return suite_gym.wrap_env(create_race_circuit_gym_env(offscreen))
