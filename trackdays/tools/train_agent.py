import argparse

from trackdays.training.train_sac_agent import train_sac_agent
from trackdays.training.utils import cudnn_workaround
from trackdays.tools.utils import create_race_circuit_tf_agents_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward-scale', metavar='VAL', type=float, required=False, default=1.0)
    parser.add_argument('--batch-size', metavar='NUM', type=int, required=False, default=32)
    parser.add_argument('--initial-collect-steps', metavar='NUM', type=int, required=False, default=100)
    parser.add_argument('--replay-buffer-size', metavar='NUM', type=int, required=False, default=100000)
    args = parser.parse_args()

    cudnn_workaround()

    train_sac_agent(
        env_factory=create_race_circuit_tf_agents_env,
        batch_size=args.batch_size,
        reward_scale_factor=args.reward_scale,
        initial_collect_steps=args.initial_collect_steps,
        replay_buffer_size=args.replay_buffer_size,
        tensorboard_dir='./tensorboard',
        checkpoint_dir='./checkpoints',
        latest_policy_dir='./policies/latest',
        best_policy_dir='./policies/best',
    )


if __name__ == '__main__':
    main()
