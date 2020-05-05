import gym
import trackdays.envs.race_circuit


def main():
    env = gym.make('racecircuit-v0')
    env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = [0, 1]
        obs, reward, done, _ = env.step(action)
        env.render()

        total_reward += reward
        print('Current total reward: {0}'.format(total_reward))


if __name__ == '__main__':
    main()
