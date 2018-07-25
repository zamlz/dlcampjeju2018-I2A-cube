
import gym
import numpy as np
import tensorflow as tf
import time

from actor_critic import RandomActorCritic
from common.model import NetworkBase
from common.multiprocessing_env import SubprocVec
from environment_model import EnvironmentModel
from imagination.core import ImaginationCore
from imagination.policy import I2ABuilder 
from tqdm import tqdm

# TODO: Finish this class
class ImaginationAugmentedAgents(NetworkBase):
    
    def __init__(self, sess, ob_space, ac_space,
                 a2c_arch, a2c_load_path, a2c_random,
                 em_arch, em_load_path):

        # Unlike the other models in the codebase, the imagination augmented agent
        # needs to have an actor critic and environment model together to work.
        # though the weights for the actor critic can be not loaded and it will
        # become a random agent.
        # -----------------------------------------------------------------------

        # Setup the Actor Critic 
        a2c = RandomActorCritic(sess, a2c_arch, ob_space, ac_space, nenvs, nsteps)

        if a2c_load_path is not None:
            a2c.load(a2c_load_path)
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

        # Now we build the imagination core, which generates batches of rollouts
        # using the actor critic model and the environment model
        self.imag_core = ImaginationCore(a2c, em, rollouts)

def train(env_fn        = None,
          spectrum      = False,
          i2a_arch      = None,
          a2c_arch      = None,
          em_arch       = None,
          nenvs         = 16,
          nsteps        = 100,
          max_iters     = 1e6,
          gamma         = 0.99,
          pg_coeff      = 1.0,
          vf_coeff      = 0.5,
          ent_coeff     = 0.01,
          max_grad_norm = 0.5,
          lr            = 7e-4,
          alpha         = 0.99,
          epsilon       = 1e-5,
          log_interval  = 100,
          summarize     = True,
          i2a_load_path = None,
          a2c_load_path = None,
          em_load_path  = None,
          log_path      = None,
          cpu_cores     = 1):

    # Construct the vectorized parallel environments
    envs = [ env_fn for _ in range(nenvs) ]
    envs = SubprocVecEnv(envs)

    # Set some random seeds for the environment
    envs.seed(0)
    if spectrum:
        envs.spectrum()

    ob_space = envs.observation_space.shape
    nw, nh, nc = ob_space
    ac_space = envs.action_space

    obs = envs.reset()

    tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=cpu_cores,
            intra_op_parallelism_threads=cpu_cores )
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # TODO: Setup the Imagination Augmented Agent
        i2a_agent = ImaginationAugmentedAgent()

        load_count = 0
        if i2a_load_path is not None:
            i2a_agent.load(i2a_load_path)

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path, graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        print('Imagination Augmented Agent Training Start!')
        print('Model will be saved on intervals of %i' % (log_interval))
        for i in tqdm(range(load_count + 1, int(max_iters)+1), ascii=True,
                      desc'ImaginationAugmentedAgent'):

            # TODO: Write the training calls here

            if summarize:
                pass
            else:
                pass

            if i % log_interval == 0:
                i2a_agent.save(log_path, i)

        i2a_agent.save(log_path, 'final')
        print('Imagination Augmented Agent is finished training')
