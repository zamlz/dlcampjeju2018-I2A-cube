
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
    for s in x.shape.to_list()[1:-1][::-1]:
        a = tf.stack([a for _ in range(s)]

    a = tf.one_hot(a, depth=n, axis=-1)
    x = tf.concat([x,a], axis=-1)
    return x




class EMBuilder(object):

    def __init__(self, sess, ob_space, ac_space, reuse=False, summarize=True):
            
            self.sess = sess

            nw, nh, nc = ob_space
            self.nact = ac_space.n

            self.ob = tf.placeholder(tf.float32, [None, nw, nh, nc])
            self.a = tf.placeholder(tf.float32, [None])

            x = self.ob

            # Construction of the network
            with tf.variable_scope("env_model", reuse=reuse
                
                x = concat_actions(x, self.a , self.nact)
            
                pred_obs = x
                pred_rew = x

            self.pred_obs = pred_obs
            self.pred_rew = pred_rew

    def predict(self, ob, a):
        return self.sess.run([self.pred_obs, self.pred_rew], {self.ob:ob, self.a:a})


def em_parser():
    pass
