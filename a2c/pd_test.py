
import time
import numpy as np
import tensorflow as tf

from a2c import ActorCritic

def pd_test(env_fn, policy, load_path):

    env = env_fn()
    actions = env.unwrapped.action_list
    env._seed(int(time.time()))

    obs = env.reset()
    obs = np.expand_dims(obs, axis=0)
    env.render()

    action_list = []

    with tf.Session() as sess:

        actor_critic = ActorCritic(sess, policy,
                env.observation_space.shape, env.action_space, 1, 5,
                0.5, 0.01, 0.5, 7e-4, 0.99, 1e-5, False)
        
        if load_path:
            actor_critic.load(load_path)
        else:
            print('WARNING: No Model Loaded!')

        d = False
        while not d:
            print('-------------------------------------------------')
            print('Current Observation')
            env.render()
            time.sleep(0.1)

            a, v, neg = actor_critic.act(obs, stochastic=True)
            print('')
            print('action: ', actions[a[0]])
            print('value: ', v)
            print('neglogp: ', neg)
            print('pd: ') 
            for ac, pd in zip(actions, actor_critic.step_model.logits(obs)[0][0]):
                print('\t', ac, pd)

            obs, r, d, _ = env.step(a)
            print('r: ', r)
            time.sleep(0.1)
            d = d[0]
            obs = np.expand_dims(obs, axis=0)

    env.close()
