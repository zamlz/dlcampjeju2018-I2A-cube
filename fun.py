
import gym
import sys
import cube_gym
import time
from common.multiprocessing_env import SubprocVecEnv
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from a2c import ActorCritic
from policy import *

def env_fn():
    env = gym.make('cube-x3-v0')
    env.unwrapped._refreshScrambleParameters(1, 2, scramble_easy=True)
    return env

actions = env_fn().unwrapped.action_list

envs = SubprocVecEnv([env_fn])

obs = envs.reset()
envs.render(0)

action_list = []

fig = plt.figure()
ims = []

im = plt.imshow(cube_gym.onehotToRGB(obs[0]))
ims.append([im])

with tf.Session() as sess:

    actor_critic = ActorCritic(sess, CnnPolicy,
            envs.observation_space.shape, envs.action_space, 1, 5,
            0.5, 0.01, 0.5, 7e-4, 0.99, 1e-5, False)
    actor_critic.load(sys.argv[1])

    # sess.run(tf.global_variables_initializer())

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

        if not d:
            im = plt.imshow(cube_gym.onehotToRGB(obs[0]))
            ims.append([im])
        else:
            print('DONE')
            im = plt.imshow(cube_gym.onehotToRGB(sbo[0]))
            ims.append([im])

        d = d[0]
        print(r)

    ani = animation.ArtistAnimation(fig, ims, interval=2000, blit=True, repeat_delay=4000)

    # plt.show()
