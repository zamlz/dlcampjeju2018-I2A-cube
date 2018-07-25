
import numpy as np
import tensorflow as tf

from common.network import NetworkBuilder
from common.network import build_conv3d, build_conv2d, build_dense, build_dense_vfp

class I2ABuilder(NetworkBuilder):

    def __init__(self, sess, build, ob_space, ac_space):
        
        self.sess = sess

        nw, nh, nc = ob_space
        self.nact = ac_space.n
        build = self.parser(build)

        self.obs = tf.placeholder(tf.float32, [None, nw, nh, nc], name='obs_original')
        self.obs_imag = tf.placeholder(tf.float32, [None, None, nw, nh, nc], name='obs_imag')
        self.rew_imag = tf.placeholder(tf.float32, [None, None], name='rew_imag' )

        x = self.obs

        with tf.variable_scope('i2a_model'):
            
            with tf.variable_scope('model_based_path'):
                xi = tf.layers.flatten(self.obs_imag)
                yi = tf.layers.flatten(self.rew_imag)
                hi = tf.concat([xi, yi], axis=1)

            with tf.variable_scope('model_free_path'):
                
                with tf.variable_scope('conv_layers'):
                    for b in build['conv3d']:
                        x = build_conv3d(x, b)
                    for b in build['conv2d']:
                        x = build_conv2d(x, b)

            x = tf.concat([x, hi], axis=1)
            
            with tf.variable_scope('hidden_layers'):
                h = build_dense(x, build['dense'][0])
            
            with tf.variable_scope('pi'):
                pi = build_dense_vfp(h, nact, build['pi'][0])

            with tf.variable_scope('v'):
                vf = build_dense_vfp(h, 1, build['vf'][0])[:, 0]

        # Sample action. `pi` is like the logits
        u = tf.random_uniform(tf.shape(pi))
        self.a0 = tf.argmax(pi- tf.log(-tf.log(u)), axis=-1)
        self.nda = tf.argmax(pi, axis=-1)

        # Get the negative log likelihood
        one_hot_actions = tf.one_hot(self.a0, pi.get_shape().as_list()[-1])
        self.neglogp0 = tf.nn.softmax_cross_entropy_with_logits(
            logits=pi,
            labels=one_hot_actions)

        self.pi = pi
        self.vf = vf

    def step(self, obs, obs_imag, rew_imag, stochastic=True):
        if stochastic:
            return self.sess.run([self.a0, self.vf, self.neglogp0],
                                 {self.obs: obs, self.obs_imag: obs_imag, self.rew_imag: rew_imag})
        else:
            return self.sess.run([self.nda, self.vf, self.neglogp0],
                                 {self.obs: obs, self.obs_imag: obs_imag, self.rew_imag: rew_imag})

    def value(self, obs, obs_imag, rew_imag):
        return self.sess.run([self.vf],
                             {self.obs: obs, self.obs_imag: obs_imag, self.rew_imag: rew_imag})

    def logits(self, obs, obs_imag, rew_imag):
        return self.sess.run([self.pi],
                             {self.obs: obs, self.obs_imag: obs_imag, self.rew_imag: rew_imag})
        
