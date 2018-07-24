
import numpy as np
import tensorflow as tf

from common.network import NetworkBuilder
from common.network import build_conv3d, build_conv2d, build_dense, build_dense_vfp

class PolicyBuilder(NetworkBuilder):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, build=None):
        
        self.sess = sess

        nw, nh, nc = ob_space
        nact = ac_space.n
        build = self.parser(build)
        assert len(build['dense']) == 1, "There should be a single dense layer"
        assert len(build['pi']) == 1, "Missing Policy Function"
        assert len(build['vf']) == 1, "Missing Value Function"

        self.X = tf.placeholder(tf.float32, [None, nw, nh, nc], name='observations') #obs
        x = self.X

        # Construction of the model
        with tf.variable_scope("a2c_model", reuse=reuse):

            with tf.variable_scope('conv_layers'):
                for b in build['conv3d']:
                    x = build_conv3d(x, b)
                for b in build['conv2d']:
                    x = build_conv2d(x, b)
            
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

    def step(self, ob, stochastic=True):
        if stochastic: 
            return self.sess.run([self.a0, self.vf, self.neglogp0], {self.X:ob})
        else:
            return self.sess.run([self.nda, self.vf, self.neglogp0], {self.X:ob})

    def value(self, ob):
        v = self.sess.run(self.vf, {self.X:ob})
        return v

    def logits(self, ob):
        pi = self.sess.run([self.pi], {self.X: ob})
        return pi

    # Next two methods are required when we will have to generate the imaginations later in the I2A
    # code.
    def transform_input(self, X):
        return [X]

    def get_inputs(self):
        return [self.X]
