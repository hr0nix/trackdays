from trackdays.tools.utils import create_race_circuit_gym_env


def main():
    env = create_race_circuit_gym_env(offscreen=False)
    env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = [0, 0]
        obs, reward, done, _ = env.step(action)
        env.render()

        total_reward += reward
        print('Current total reward: {0}'.format(total_reward))


if __name__ == '__main__':
    main()
