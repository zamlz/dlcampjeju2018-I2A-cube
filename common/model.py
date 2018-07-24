
import os
import numpy as np

# A generator for stepping the agent with the environment
def model_play_games(model, envs, nsteps):
    s = envs.reset()
    for i in range(nsteps):
        a, _, _ = model.act(s)
        ns, r, d, info = envs.step(a)
        true_ns = [ x['obs'] for x in info ]
        true_ns = np.stack(true_ns, axis=0)
        yield s, a, r, true_ns, d
        s = ns

# Base Class for various network archictures
class NetworkBase(object):

    def __init__(self):
        pass

    def train(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

    def critique(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save(self, path, step):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, path + str(step) + '.ckpt')

    def load(self, full_path):
        self.saver.restore(self.sess, full_path)
