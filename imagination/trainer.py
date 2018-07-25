
import gym
import numpy as np
import tensorflow as tf
import time

from common.model import NetworkBase
from common.multiprocessing_env import SubprocVec
from imagination.core import ImaginationCore
from imagination.policy import I2ABuilder 
from tqdm import tqdm

# Much of the code here is similar to the actor critic
# training code. Thats because it is still trained
# the same way, just we're using a different model
# and with different hyperparameters
# If I was smart, I would have modularized it
class ImaginationAugmentedAgents(NetworkBase):
    
    def __init__(self, sess, i2a_arch, ob_space, ac_space, a2c_arch, a2c_load_path,
                 em_arch, em_load_path, horizon,
                 pg_coeff=1.0, vf_coeff=0.5, ent_coeff=0.01, max_grad_norm=0.5,
                 lr=7e-4, alpha=0.99, epsilon=1e-5, summarize=False):

        self.sess = sess
        self.nact = ac_space.n
        self.ob_space = ob_space

        # Actions Advantages and Reward
        self.actions = tf.placeholder(tf.int32, [None], name='actions')
        self.advantages = tf.placeholder(tf.float32, [None], name='advantages')
        self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
        self.depth = tf.placeholder(tf.float32, [None], name='scramble_depth')

        # We build the imagination core, which generates batches of rollouts
        # using the actor critic model and the environment model
        self.core = ImaginationCore(sess, ob_space, ac_space, a2c_arch, a2c_load_path,
                                    em_arch, em_load_path, horizon)
        # And setup the model
        self.step_model = I2ABuilder(sess, i2a_arch, ob_space, ac_space, reuse=False)
        self.train_model = I2ABuilder(sess, i2a_arch, ob_space, ac_space, reuse=True)

        # Policy Gradients Loss, Value Function Loss, Entropy, and Full Loss
        self.pg_loss = tf.reduce_mean(self.advantages * neglogpac)
        self.vf_loss = tf.reduce_mean(tf.square(tf.squeeze(self.train_model.vf) - self.rewards) / 2.0)
        self.entropy = tf.reduce_mean(cat_entropy(self.train_model.pi))
        self.loss = pg_coeff*self.pg_loss - ent_coeff*self.entropy + vf_coeff*self.vf_loss
        
        self.mean_rew= tf.reduce_mean(self.rewards)
        self.mean_depth = tf.reduce_mean(self.depth)

        # Find the model parameters and their gradients
        with tf.variable_scope('i2a_model'):
            self.params = tf.trainable_variables()
        grads = tf.gradients(self.loss, self.params)

        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, self.params))

        # Setup the optimizer
        trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
        self.opt = trainer.apply_gradients(grads)

        # For some awesome tensorboard stuff
        if summarize:
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Entropy', self.entropy)
            tf.summary.scalar('Policy Gradient Loss', self.pg_loss)
            tf.summary.scalar('Value Function Loss', self.vf_loss)
            tf.summary.scalar('Rewards', self.mean_rew)
            tf.summary.scalar('Depth', self.mean_depth)

        # Initialize the tensorflow saver
        self.saver = tf.train.Saver(self.params, max_to_keep=5)


def train(env_fn        = None,
          spectrum      = False,
          i2a_arch      = None,
          a2c_arch      = None,
          em_arch       = None,
          nenvs         = 16,
          nsteps        = 100,
          horizon       = 1,
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
        i2a_agent = ImaginationAugmentedAgent(sess, i2a_arch, ob_space, ac_space,
                                              a2c_arch, a2c_load_path,
                                              em_arch, em_load_path,
                                              horizon)

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
