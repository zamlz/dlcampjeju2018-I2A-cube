# Inspired from OpenAI Baselines. This uses the same design of having an easily
# substitutable generic policy that can be trained. This allows to easily
# substitute in the I2A policy as opposed to the basic CNN one.
import os
import numpy as np
import tensorflow as tf


# Basic baseline policy
class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        nw, nh, nc = ob_space

        nact = ac_space.n
        X = tf.placeholder(tf.float32, [None, nw, nh, nc]) #obs
        with tf.variable_scope("model", reuse=reuse):
            conv1 = tf.layers.conv2d(activation=tf.nn.relu,
                                        inputs=X,
                                        filters=16,
                                        kernel_size=[3,3],
                                        strides=[1,1],
                                        padding='VALID')
            conv2 = tf.layers.conv2d(activation=tf.nn.relu,
                                        inputs=conv1,
                                        filters=16,
                                        kernel_size=[3,3],
                                        strides=[2,2],
                                        padding='VALID')
            h = tf.layers.dense(tf.layers.flatten(conv2), 256, activation=tf.nn.relu)
            with tf.variable_scope('pi'):
                pi = tf.layers.dense(h, nact,
                        activation=None,
                        kernel_initializer=tf.random_normal_initializer(0.01),
                        bias_initializer=None)

            with tf.variable_scope('v'):
                vf = tf.layers.dense(h, 1,
                        activation=None,
                        kernel_initializer=tf.random_normal_initializer(0.01),
                        bias_initializer=None)[:, 0]

        # Sample action. `pi` is like the logits
        u = tf.random_uniform(tf.shape(pi))
        self.a0 = tf.argmax(pi- tf.log(-tf.log(u)), axis=-1)

        # Get the negative log likelihood
        one_hot_actions = tf.one_hot(self.a0, pi.get_shape().as_list()[-1])
        self.neglogp0 = tf.nn.softmax_cross_entropy_with_logits(
            logits=pi,
            labels=one_hot_actions)

        self.X = X
        self.pi = pi
        self.vf = vf

    def step(self, sess, ob):
        a, v, neglogp = sess.run([self.a0, self.vf, self.neglogp0], {self.X:ob})
        return a, v, neglogp

    def value(self, sess, ob):
        v = sess.run(self.vf, {self.X:ob})
        return v

    # Next two methods are required when we will have to generate the imaginations later in the I2A
    # code.
    def transform_input(self, X, sess):
        return [X]

    def get_inputs(self):
        return [self.X]

