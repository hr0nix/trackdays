import gym
import trackdays.envs.race_circuit


def main():
    env = gym.make('racecircuit-v0')
    env.reset()
    done = False
    while not done:
        action = [0, 0]
        obs, reward, done, _ = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
