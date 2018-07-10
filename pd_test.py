
import gym
import sys
import cube_gym
import time
from common.multiprocessing_env import SubprocVecEnv
import tensorflow as tf

from a2c import ActorCritic
from policy import *

def env_fn():
    env = gym.make('cube-x3-v0')
    env.unwrapped._refreshScrambleParameters(1, 2)
    return env

actions = env_fn().unwrapped.action_list

envs = SubprocVecEnv([env_fn])

obs = envs.reset()
envs.render(0)

action_list = []

with tf.Session() as sess:

    actor_critic = ActorCritic(sess, CnnPolicy,
            envs.observation_space.shape, envs.action_space, 1, 5,
            0.5, 0.01, 0.5, 7e-4, 0.99, 1e-5, False)
    actor_critic.load(sys.argv[1])

    d = False
    while not d:
        print('-------------------------------------------------')
        print('Current Observation')
        envs.render(0)
        time.sleep(0.1)

        a, v, neg = actor_critic.act(obs, stochastic=True)
        print('')
        print('action: ', actions[a[0]])
        print('value: ', v)
        print('neglogp: ', neg)
        print('pd: ') 
        for ac, pd in zip(actions, actor_critic.step_model.logits(obs)[0][0]):
            print('\t', ac, pd)

        obs, r, d, sbo = envs.step(a)
        print('r: ', r)
        envs.render(0)
        time.sleep(0.1)

        d = d[0]
        print(r)
