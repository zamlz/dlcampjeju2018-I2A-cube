
import os
import numpy as np
import tensorflow as tf

from actor_critic.policy import build_conv3d, build_conv2d, build_dense

# Given some tensor x, concatenate the
# one-hot expanded of a with depth n to
# x and return it
# x : [None, d1, d2, ..., dn, c]
# a : [None]
# xa: [None, d1, d2, ..., dn, c+n]
def concat_actions(x, a, n):
 
    # Find other shapes to expand to
    a = tf.one_hot(a, depth=n, axis=-1)
    for s in x.shape.as_list()[1:-1][::-1]:
        a = tf.stack([a for _ in range(s)], axis=1)
    x = tf.concat([x,a], axis=-1)
    return x


class EMBuilder(object):

    def __init__(self, sess, build, ob_space, ac_space, summarize=True):
            
            self.sess = sess

            nw, nh, nc = ob_space
            self.nact = ac_space.n

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


def em_parser(builder):
    builder = builder.split('_')
    builder = [ b.split(":") for b in builder ]
    bd = {
        'conv3d' : [],
        'conv2d' : [],
        'dense': [],
    }
    for b in builder:
        if 'c3d' in b[0]:
            bd['conv3d'].append(b)
        elif 'c2d' in b[0]:
            bd['conv2d'].append(b)
        elif 'h' in b[0]:
            bd['dense'].append(b)
        else:
            pass
    return bd


