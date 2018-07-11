
import time
import tensorflow as tf

from a2c import ActorCritic
from common.multiprocessing_env import SubprocVecEnv

def pd_test(env_fn, policy, load_path):
    actions = env_fn().unwrapped.action_list

    envs = SubprocVecEnv([env_fn])
    envs.seed(int(time.time()))

    obs = envs.reset()
    envs.render(0)

    action_list = []

    with tf.Session() as sess:

        actor_critic = ActorCritic(sess, policy,
                envs.observation_space.shape, envs.action_space, 1, 5,
                0.5, 0.01, 0.5, 7e-4, 0.99, 1e-5, False)
        actor_critic.load(load_path)

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

            obs, r, d, _ = envs.step(a)
            print('r: ', r)
            time.sleep(0.1)

            d = d[0]
            print(r)

    envs.close()