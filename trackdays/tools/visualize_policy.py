import argparse

from trackdays.training.utils import load_policy, visualize_policy, as_tf_env
from trackdays.tools.utils import create_race_circuit_tf_agents_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-path', metavar='FILE', type=str, required=True)
    parser.add_argument('--output-file', metavar='FILE', type=str, required=True)
    parser.add_argument('--episodes', metavar='NUM', type=int, required=False, default=1)
    parser.add_argument('--fps', metavar='NUM', type=int, required=False, default=1)
    args = parser.parse_args()

    policy = load_policy(args.policy_path)
    env = as_tf_env(create_race_circuit_tf_agents_env())
    visualize_policy(env, policy, args.output_file, args.episodes)


if __name__ == '__main__':
    main()
