
import numpy as np
import tensorflow as tf

from actor_critic import RandomActorCritic
from environment_model import EnvironmentModel

# This is a the primary workhorse of the 
# I2A model. It generates batches of 
# trajectories using the environment model
# and the random actor critic agent

class ImaginationCore(object):

    def __init__(self, sess, ob_space, ac_space, a2c_arch, a2c_load_path, 
                 em_arch, em_load_path, horizon):
 
        # Unlike the other models in the codebase, the imagination augmented agent
        # needs to have an actor critic and environment model together to work.
        # though the weights for the actor critic can be not loaded and it will
        # become a random agent.
        # -----------------------------------------------------------------------

        # Setup the Actor Critic 
        a2c = RandomActorCritic(sess, a2c_arch, ob_space, ac_space)

        if a2c_load_path is not None:
            a2c.load(a2c_load_path)
            a2c.epsilon = a2c_random
            with open(logpath+'/a2c_load_path'. 'w') as a2cfile:
                a2cfile.write(a2c_load_path)
            print('Loaded Actor Critic Weights')
        else:
            a2c.epsilon = -1
            print('WARNING: No Actor Critic Model loaded. Using Random Agent')

        # Setup the Environment Model
        em = EnvironmentModel(sess, em_arch, ob_space, ac_space)

        if em_load_path is not None:
            em.load(em_load_path)
            with open(logpath+'/em_load_path'. 'w') as emfile:
                emfile.write(em_load_path)
            print('Loaded Environment Model Weights')
        else:
            print('WARNING: No Environment Model loaded. Using empty rollouts')
       
        self.a2c = a2c
        self.em = em

        # The initial action for each rollout will be unique,
        # it will follow the actor critic policy there after.
        # That is why the rollnum is set to the size of the
        # action space
        self.a_n = ac_space.n
        self.horizon = horizon

    def imagine(self, obs):

        # Batch Size, Width, Height, Channels
        bs, nw, nh, nc = obs.shape

        # Really do yourself a favor here and numpy+ipython the shit outta this
        # Its really complicated to visualize >.<
        # But i'll try my best to explain
        # Imagine that this is the original batch data of true observations
        # 
        #   [ob_1, ob_2, ob_3]
        #   (apply tiling based on actionspace [a_n = 2 here])
        #   [[ob_1, ob_2, ob_3], [ob_1, ob_2, ob_3]]
        #   (apply reshape)
        #   [ob_1, ob_2, ob_3, ob_1, ob_2, ob_3]
        #
        # Now we create its corresponding action values
        # This could have been made with python list creation stuff
        # but i was worried it would be a bit slow and numpy uses
        # C under the hood so I think? its faster??? (need to verify)
        #
        #   [0, 1]
        #   (apply reshape)
        #   [[0],[1]]
        #   (apply tiling based on batchsize)
        #   [[0,0,0],[1,1,1]]
        #   (apply final reshape)
        #   [0, 0, 0, 1, 1, 1]
        # 
        # And we're done, we now have the initial observation and action
        # vectors in a form suitable for the environment model and the
        # actor critic

        obs = np.tile(obs, [self.a_n, 1, 1, 1, 1]).reshape([-1, nw, nh, nc])
        a = np.tile(np.arange(self.a_n).reshape([-1,1]), bs).reshape([-1])

        # Rollout Batch Size (the new 0 dim of the obs and actions)
        rbs = bs * self.a_n
        # Rollout Batches
        obs_rb, rew_rb = [], []

        # Now we shall hallucinate, wait no I mean imagine
        for step in range(self.horizon):
           
            # Pass the observation and actions to the environment model
            obs_imag, rew_imag = self.em.predict(obs, a)
            
            # TODO: clean imagined observations and reward (optional)
           
            # Add them to the full rollout batch
            obs_rb.append(obs_imag)
            rew_rb.append(rew_imag)
    
            # Update vars for the next timestep
            obs = obs_imag
            a, _, _ = self.a2c.act(obs)

        return np.array(obs_rb), np.array(rew_rb)

