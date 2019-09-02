import argparse
import gym_fem

import gym
from gym import wrappers, logger
from gym_fem.fem_env import FEMCSVLogger #, PseudoContinuousActions


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='2d-deepdrawing-5ts-v2', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, video_callable=False, write_upon_reset=True, force=True)
    env = FEMCSVLogger(env, outdir=outdir)
    # env = PseudoContinuousActions(env)
    # env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 10000
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    # Close the env and write monitor result info to disk
    env.close()
