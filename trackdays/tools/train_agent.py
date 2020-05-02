import trackdays.envs.race_circuit

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

import gym

from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import actor_distribution_network, normal_projection_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import greedy_policy, random_tf_policy
from tf_agents.utils import common


def load_env(env_config):
    gym_env = gym.make('racecircuit-v0', config=env_config or {})
    tf_agents_env = suite_gym.wrap_env(gym_env)
    return tf_py_environment.TFPyEnvironment(tf_agents_env)


def create_critic_network(train_env):
    return critic_network.CriticNetwork(
        (train_env.observation_spec(), train_env.action_spec()),
        observation_conv_layer_params=[
            (5, 3, 2),
            (10, 3, 2),
            (15, 3, 2),
            (20, 3, 2),
            (25, 3, 2),
        ],
        observation_fc_layer_params=[32],
        action_fc_layer_params=[4, 16, 32],
        joint_fc_layer_params=[8],
    )


def create_actor_network(train_env):
    def projection_net_factory(action_spec):
        return normal_projection_network.NormalProjectionNetwork(
            action_spec,
            mean_transform=None,
            state_dependent_std=True,
            init_means_output_factor=0.1,
            std_transform=sac_agent.std_clip_transform,
            scale_distribution=True,
        )

    return actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        conv_layer_params=[
            (5, 3, 2),
            (10, 3, 2),
            (15, 3, 2),
            (20, 3, 2),
            (25, 3, 2),
        ],
        fc_layer_params=[8],
        continuous_projection_net=projection_net_factory,
    )


def create_sac_agent(train_env):
    return sac_agent.SacAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=create_actor_network(train_env),
        critic_network=create_critic_network(train_env),
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=3e-4),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=3e-4),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=3e-4),
        target_update_tau=0.005,
        target_update_period=1,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
    )


def compute_average_return(environment, policy, num_episodes):
    print('Evaluating agent')

    total_return = 0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    return (total_return / num_episodes).numpy()[0]


def create_replay_buffer(agent, train_env, replay_buffer_size):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_size,
    )


def create_collect_driver(train_env, agent, replay_buffer, collect_steps):
    return dynamic_step_driver.DynamicStepDriver(
        train_env, agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps,
    )


def train_agent(
        batch_size=32,
        total_training_steps=100000,
        loss_report_rate=100,
        avg_return_report_rate=500,
        initial_collect_steps=10000,
        training_iteration_collect_steps=1,
        replay_buffer_size=10000,
        num_eval_episodes=3,
        env_config=None,
):
    train_env = load_env(env_config)
    eval_env = load_env(env_config)

    agent = create_sac_agent(train_env)
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    eval_policy = greedy_policy.GreedyPolicy(agent.policy)

    replay_buffer = create_replay_buffer(agent, train_env, replay_buffer_size)

    collect_driver = create_collect_driver(
        train_env, agent, replay_buffer, collect_steps=training_iteration_collect_steps)
    collect_driver.run = common.function(collect_driver.run)

    initial_collect_driver = create_collect_driver(
        train_env, agent, replay_buffer, collect_steps=initial_collect_steps
    )
    initial_collect_driver.run()

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=2, sample_batch_size=batch_size, num_steps=2
    ).prefetch(1)
    dataset_iter = iter(dataset)

    avg_return = compute_average_return(eval_env, eval_policy, num_episodes=num_eval_episodes)
    print('Before start: avg return={0}'.format(avg_return))

    for _ in range(total_training_steps):
        collect_driver.run()

        experience, _ = next(dataset_iter)
        train_loss = agent.train(experience)
        step = agent.train_step_counter.numpy()
        if step % loss_report_rate == 0:
            print('Step {0}: loss={1}'.format(step, train_loss.loss))

        if step % avg_return_report_rate == 0:
            avg_return = compute_average_return(eval_env, eval_policy, num_episodes=num_eval_episodes)
            print('Step {0}: avg return={1}'.format(step, avg_return))

    return agent


if __name__ == '__main__':
    train_agent()
