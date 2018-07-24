
import os
import numpy as np
import tensorflow as tf

from common.network import NetworkBuilder
from common.network import build_conv3d, build_conv2d, build_dense, concat_actions

class EMBuilder(NetworkBuilder):

    def __init__(self, sess, build, ob_space, ac_space):
            
            self.sess = sess

            nw, nh, nc = ob_space
            self.nact = ac_space.n
            build = self.parser(build)

            self.obs = tf.placeholder(tf.float32, [None, nw, nh, nc], name='observations')
            self.a = tf.placeholder(tf.uint8, [None], name='actions')

            x = self.obs

            # Construction of the network
            with tf.variable_scope("env_model"): 

                with tf.variable_scope('action_to_onehot_channel'):
                    x = concat_actions(x, self.a, self.nact)

                with tf.variable_scope('conv_layers'):
                    for b in build['conv3d']:
                        x = build_conv3d(x, b)
                    for b in build['conv2d']:
                        x = build_conv2d(x, b)

                with tf.variable_scope('hidden_layers'):
                    h = build_dense(x, build['dense'][0])

                with tf.variable_scope('pred_observation'):
                    pred_obs = tf.layers.dense(h, nw*nh*nc, activation=tf.nn.relu)
                    self.pred_obs = tf.reshape(pred_obs, [-1, nw, nh, nc])
               
                with tf.variable_scope('pred_reward'):
                    pred_rew = tf.layers.dense(h, 1, activation=tf.nn.relu)
                    self.pred_rew = tf.reshape(pred_rew, [-1])


    def predict(self, ob, a):
        return self.sess.run([self.pred_obs, self.pred_rew], {self.obs:ob, self.a:a})

