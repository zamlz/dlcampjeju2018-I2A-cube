

import tensorflow as tf
import numpy as np

from common.network import NetworkBuilder
from common.network import build_conv3d, build_conv2d, concat_actions, sample_z

class VAEBuilder(NetworkBuilder):
    
    def __init__(self, sess, build, ob_space, ac_space):

        nw, nh, nc = ob_space
        nact = ac_space.n
        build = self.parser(build)
        assert len(bd['en_z_de']) == 1

        self.obs = tf.placeholder(tf.float32, [None, nw, nh, nc], name='observations')
        self.a = tf.placeholder(tf.uint8, [None], name='actions')

        x = self.obs
        
        with tf.variable_scope('vae_model'):

            with tf.variable_scope('encoder'):
                for b in build['conv3d']:
                    x = build_conv3d(x, b)
                for b in build['conv2d']:
                    x = build_conv2d(x, b)

            x = tf.layers.flatten(x)
            self.z_mu     = tf.layers.dense(x, units=build['en_z_de'][0][1], name='z_mu')
            self.z_logvar = tf.layers.dense(x, units=build['en_z_de'][0][1], name='z_logvar')
            self.z        = sample_z(self.z_mu, self.z_logvar)
            x = tf.layers.dense(self.z, units=build['en_z_de'][0][2], activation=None)
            x = tf.reshape(x, [-1, 1, 1, build['en_z_de'][0][2] ] )
            
            with tf.variable_scope('decoder'):
                for b in build['conv2dT']:
                    x = build_conv2d(x, b)
                for b in build['conv3dT']:
                    x = build_conv3d(x, b)

            self.reconstructions = x

