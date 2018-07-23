
import os
import gym
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from environment_model.network import EMBuilder, em_parser
from environment_model.util import RandomActorCritic
from common.multiprocessing_env import SubprocVecEnv


class EnvironmentModel(object):

    def __init__(self, sess, em_arch, ob_space, ac_space, obs_coeff, rew_coeff, summarize):
        
        self.sess = sess
        self.nact = ac_space.n
        nw, nh, nc = ob_space

        # Setup targets
        self.target_obs = tf.placeholder(tf.uint8, [None, nw, nh, nc])
        self.target_rew = tf.placeholder(tf.uint8, [None])

        # Setup the Graph for the Environment Model
        self.model = EMBuilder(sess, ob_space, ac_space, reuse=True, summarize=summarize)
         
        # Compute the losses
        self.obs_loss = tf.losses.softmax_cross_entropy(self.target_obs, self.model.pred_obs)
        self.rew_loss = tf.losses.softmax_cross_entropy(self.target_rew, self.model.pred_rew)
        self.loss = (obs_coeff*obs_loss) + (rew_coeff*rew_loss)

        # Find the model parameters
        with tf.variable_scope('env_model'):
            self.params = tf.trainable_variables()
        grads = tf.gradients(self.loss, self.params)
        grads = list(zip(grads, self.params))

        # Setup the optimizer
        trainer = tf.train.AdamOptimizer()
        self.opt = trainer.apply_gradients(grads)

        if summarize:
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Observation Loss', self.obs_loss)
            tf.summary.scalar('Reward Loss', self.rew_loss)

        self.saver = tf.train.Saver(params, max_to_keep=100000000)

    # Single training step
    def train(self, obs, actions, tar_obs, tar_rew, summary_op=None):
        feed_dict = {
            self.model.obs: obs,
            self.model.a: actions,
            self.target_obs: tar_obs,
            self.target_rew: tar_rew,
        }

        ret_vals = [
            self.loss,
            self.obs_loss,
            self.rew_loss,
            self.opt,
        ]

        if summary_op is not None:
            ret_vals.append(summary_ops)
        
        return self.sess.run(ret_vals, feed_dict=feed_dict)

    # Given an observation and an action, return the predicted next observation and reward
    def predict(obs, a):
        return self.model.predict(obs, a)

    # Dump the model parameters in the specified path
    def save(self, path, step):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, path + str(step) + '.ckpt')

    # Load a pretrained model
    def load(self, full_path):
        self.saver.restore(self.sess, full_path)


def train(env_fn=None,
          spectrum = False,
          em_arch=None,
          a2c_policy=None,
          nenvs = 16,
          nsteps = 100,
          max_iters = 1e6,
          lr = 7e-4,
          log_interval = 1000,
          load_count = 0,
          summarize=True,
          em_load_path=None,
          a2c_load_path=None,
          log_path=None,
          cpu_cores=1):

    # Construct the vectorized parallel environments
    envs = [ env_fn for _ in range(nenvs) ]
    envs = SubprocVecEnvs(envs)

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

        actor_critic = RandomActorCritic(sess, a2c_policy, ob_space, ac_space, nenvs, nsteps)

        if a2c_load_path is not None:
            actor_critic.load(a2c_load_path)
            print('Loaded a2c')
        else:
            actor_critic.epsilon = -1
            print('WARNING: No Actor Critic Model loaded. Using Random Agent')

        env_model = EnvironmentModel(sess, em_arch, ob_space, ac_space, obs_coeff, rew_coeff, summarize)

        if em_load_path is not None:
            env_model.load(em_load_path)

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path, graph=sess.graph)

        sess.run(tf.global_variables_initializer())

        print('Env Model Training Start!')
        print('Model will train for  %i iterations' % (max_iters))
        for i, s, a, r, ns, d in tqdm(play_games(actor_critic, envs, max_iters), total=max_iters):
           
            if summarize:
                writer.add_summary(summary, i)

            if i % log_interval == 0:
                env_model.save(log_path, i)
