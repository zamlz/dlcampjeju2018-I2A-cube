
import os
import numpy as np
import tensorflow as tf


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

    def __init__(self, sess, ob_space, ac_space, summarize=True):
            
            self.sess = sess

            nw, nh, nc = ob_space
            self.nact = ac_space.n

            self.obs = tf.placeholder(tf.float32, [None, nw, nh, nc], name='observations')
            self.a = tf.placeholder(tf.uint8, [None], name='actions')

            x = self.obs

            # Construction of the network
            with tf.variable_scope("env_model"): 
                
              
                # Terrible testing code
                x = tf.layers.flatten(x)
                x = concat_actions(x, self.a , self.nact)
                x = tf.layers.dense(x, 4096, activation=tf.nn.relu)
                x = tf.layers.dense(x, 4096, activation=tf.nn.relu)
                x = tf.layers.dense(x, 4096, activation=tf.nn.relu)
            
                pred_rew = tf.layers.dense(x, 1, activation=tf.nn.relu)
                pred_obs = tf.layers.dense(x, nw*nh*nc, activation=tf.nn.relu)

                self.pred_obs = tf.reshape(pred_obs, [-1, nw, nh, nc])
                self.pred_rew = tf.reshape(pred_rew, [-1])

    def predict(self, ob, a):
        return self.sess.run([self.pred_obs, self.pred_rew], {self.obs:ob, self.a:a})


def em_parser():
    pass
