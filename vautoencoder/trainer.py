
import gym
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from vautoencoder.network import VAEBuilder
from common.multiprocessing_env import SubprocVecEnv
from common.model import NetworkBase

class VariationalAutoEncoder(NetworkBase):
    
    def __init__(self, sess, var_arch, ob_space, ac_space, lr=0.001, kl_coeff=0.5, summarize=False):

        self.sess = sess
        self.nact = ac_space.n
        nw, nh, nc = ob_space

        # Setup targets
        self.target_obs = tf.placeholder(tf.float32, [None, nw, nh, nc], name='target_obs')

        # Setup the network
        self.vae = VAEBuilder(sess, var_arch, ob_space, ac_space, summarize=summarize)

        # Compute losses
        logits_flat = tf.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.target_obs)

        self.reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis=1)
        self.kl_loss = tf.exp(self.var.z_logvar) + self.vae.z_mu**2 - 1 - self.var.z_logvar
        self.kl_loss = kl_coeff*tf.reduce_sum(self.kl_loss, 1)
        self.loss = tf.reduce_mean(self.reconstruction_loss + self.kl_loss)

        # Find the model parameters
        with tf.variable_scope('vae_model'):
            self.params = tf.trainable_variables()
        grads = tf.gradients(self.loss, self.params)
        grads = list(zip(grads, self.params))

        # Setup the optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=lr)
        self.opt = trainer.apply_gradients(grads)

        if summarize:
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('KL Loss', self.kl_loss)
            tf.summary.scalar('Reconstruction Loss', self.reconstruction_loss)

        self.saver = tf.train.Saver(self.params, max_to_keep=5)


    def train(self, obs, actions, tar_obs, tar_rew, summary_op=None):
        
        feed_dict = {
            self.vae.obs: obs,
            self.var.a: actions,
            self.target_obs: tar_obs,
            self.target_rew: tar_rew,
        }

        ret_vals = [
            self.loss,
            self.reconstruction_loss,
            self.kl_loss,
            self.opt,
        ]

        if summary_op is not None:
            ret_vals.append(summary_op)

        return self.sess.run(ret_vals, feed_dict=feed_dict)
