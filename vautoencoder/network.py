

import tensorflow as tf
import numpy as np

# The encoder block
def encoder(x):
    x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2,
                         padding='valid', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2,
                         padding='valid', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2,
                         padding='valid', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2,
                         padding='valid', activation=tf.nn.relu)
    
    x = tf.layers.flatten(x)
    z_mu     = tf.layers.dense(x, units=32, name='z_mu')
    z_logvar = tf.layers.dense(x, units=32, name='z_logvar')

    return z_mu, z_logvar

# The decoder block
def decoder(x):
    x = tf.layers.dense(x, 1024, activation=None)
    x = tf.reshape(x, [-1, 1, 1, 1024])
    x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2,
                                   padding='valid', activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2,
                                   padding='valid', activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=2,
                                   padding='valid', activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, filters=6, kernel_size=6, strides=2,
                                   padding='valid', activation=tf.nn.relu)

def sample_z(self, mu, logvar):
    eps = tf.random.normal(tf.shape(mu))
    return mu + tf.exp(logvar / 2) * eps



class VAEBuilder(object):
    
    def __init__(self, sess, var_arch, ob_space, ac_space):

        nw, nh, nc = ob_space
        nact = ac_space.n

        self.obs = tf.placeholder(tf.float32, [None, nw, nh, nc], name='obs')
        
        with tf.variable_scope('vae_model'):
            self.z_mu, self.z_logvar    = encoder(self.obs)
            self.z                      = sample_z(self.z_mu, self.z_logvar)
            self.reconstructions        = decoder(self.z)
