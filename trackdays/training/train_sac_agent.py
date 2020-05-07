import tempfile

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from trackdays.training.utils import create_replay_buffer, create_collect_driver, as_tf_env, evaluate_policy

from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network, normal_projection_network
from tf_agents.policies import greedy_policy
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils import common


def create_critic_network(train_env):
    return critic_network.CriticNetwork(
        (train_env.observation_spec(), train_env.action_spec()),
        observation_conv_layer_params=[
            (4, (5, 1), 1),
            (4, (1, 5), 2),
            (8, (5, 1), 1),
            (8, (1, 5), 2),
            (16, (5, 1), 1),
            (16, (1, 5), 2),
            (32, (5, 1), 1),
            (32, (1, 5), 2),
        ],
        action_fc_layer_params=[128],
        joint_fc_layer_params=[128, 128],
    )


def create_actor_network(train_env):
    def projection_net_factory(action_spec):
        return normal_projection_network.NormalProjectionNetwork(
            action_spec,
            mean_transform=None,
            state_dependent_std=True,
            std_transform=sac_agent.std_clip_transform,
            scale_distribution=True
        )

    return actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        conv_layer_params=[
            (4, (5, 1), 1),
            (4, (1, 5), 2),
            (8, (5, 1), 1),
            (8, (1, 5), 2),
            (16, (5, 1), 1),
            (16, (1, 5), 2),
            (32, (5, 1), 1),
            (32, (1, 5), 2),
        ],
        fc_layer_params=[128, 128],
        continuous_projection_net=projection_net_factory,
    )


def create_sac_agent(train_env, reward_scale_factor):
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
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=None,
        train_step_counter=tf.compat.v1.train.get_or_create_global_step(),
    )


def train_sac_agent(
        env_factory,
        batch_size=128,
        reward_scale_factor=1.0,
        total_training_steps=1000000,
        eval_callback_rate=None,
        eval_callback=None,
        avg_return_report_rate=1000,
        initial_collect_steps=10000,
        training_iteration_collect_steps=1,
        replay_buffer_size=120000,
        num_eval_episodes=3,
        checkpoint_dir=None,
        latest_policy_dir=None,
        best_policy_dir=None,
        tensorboard_dir=None,
        latest_policy_save_rate=5000,
        checkpoint_save_rate=20000,
):
    train_env = as_tf_env(env_factory())
    eval_env = as_tf_env(env_factory())

    agent = create_sac_agent(train_env, reward_scale_factor)
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

    if checkpoint_dir is None:
        checkpoint_dir = tempfile.mkdtemp()
    print('Checkpoints will be stored in {0}'.format(checkpoint_dir))
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=agent.train_step_counter,
    )
    train_checkpointer.initialize_or_restore()

    if latest_policy_dir is None:
        latest_policy_dir = tempfile.mkdtemp()
    print('Learned policies will be stored in {0}'.format(latest_policy_dir))
    latest_policy_saver = PolicySaver(eval_policy)

    if best_policy_dir is None:
        best_policy_dir = tempfile.mkdtemp()
    print('Learned policies will be stored in {0}'.format(best_policy_dir))
    best_policy_saver = PolicySaver(eval_policy)

    if tensorboard_dir is None:
        tensorboard_dir = tempfile.mkdtemp()
    print('Tensorboard logs will be stored in {0}'.format(tensorboard_dir))
    writer = tf.summary.create_file_writer(tensorboard_dir)

    with writer.as_default():
        avg_return, avg_num_steps = evaluate_policy(eval_env, eval_policy, num_episodes=num_eval_episodes)
        tf.summary.scalar('Average return', avg_return, step=0)
        tf.summary.scalar('Average number of steps', avg_num_steps, step=0)
        best_avg_return = avg_return

        for _ in range(total_training_steps):
            collect_driver.run()

            experience, _ = next(dataset_iter)
            agent.train(experience)
            step = agent.train_step_counter.numpy()

            if step % avg_return_report_rate == 0:
                avg_return, avg_num_steps = evaluate_policy(eval_env, eval_policy, num_episodes=num_eval_episodes)
                tf.summary.scalar('Average return', avg_return, step=step)
                tf.summary.scalar('Average number of steps', avg_num_steps, step=step)

                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    best_policy_saver.save(best_policy_dir)

            if eval_callback_rate is not None and step % eval_callback_rate == 0:
                eval_callback(eval_env, eval_policy)

            if latest_policy_save_rate is not None and step % latest_policy_save_rate == 0:
                latest_policy_saver.save(latest_policy_dir)

            if checkpoint_save_rate is not None and step % checkpoint_save_rate == 0:
                train_checkpointer.save(agent.train_step_counter)

    return agent
