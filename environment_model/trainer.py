
import gym
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from environment_model.network import EMBuilder
from actor_critic import RandomActorCritic
from common.multiprocessing_env import SubprocVecEnv
from common.model import NetworkBase, model_play_games


class EnvironmentModel(NetworkBase):

    def __init__(self, sess, em_arch, ob_space, ac_space, loss_fn='mse', lr=0.001,
                 obs_coeff=1.0, rew_coeff=1.0, summarize=False):
        
        self.sess = sess
        self.nact = ac_space.n
        nw, nh, nc = ob_space

        # Setup targets
        self.target_obs = tf.placeholder(tf.float32, [None, nw, nh, nc], name='target_observations')
        self.target_rew = tf.placeholder(tf.float32, [None], name='target_rewards')

        # Setup the Graph for the Environment Model
        self.model = EMBuilder(sess, em_arch, ob_space, ac_space)
         
        # Compute the losses (defaults to MSE)
        if loss_fn is 'ent':
            self.obs_loss = tf.losses.softmax_cross_entropy(self.target_obs, self.model.pred_obs)
            self.rew_loss = tf.losses.softmax_cross_entropy(self.target_rew, self.model.pred_rew)
        else:
            self.obs_loss = tf.reduce_mean(tf.square(self.target_obs - self.model.pred_obs) / 2.0)
            self.rew_loss = tf.reduce_sum(tf.square(self.target_rew - self.model.pred_rew) / 2.0)
        self.loss = (obs_coeff*self.obs_loss) + (rew_coeff*self.rew_loss)

        # Find the model parameters
        with tf.variable_scope('env_model'):
            self.params = tf.trainable_variables()
        grads = tf.gradients(self.loss, self.params)
        grads = list(zip(grads, self.params))

        # Setup the optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=lr)
        self.opt = trainer.apply_gradients(grads)

        if summarize:
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Observation Loss', self.obs_loss)
            tf.summary.scalar('Reward Loss', self.rew_loss)

        self.saver = tf.train.Saver(self.params, max_to_keep=5000000)

    # Single training step
    def train(self, obs, actions, tar_obs, tar_rew, summary_op=None):
        feed_dict = {
            self.model.obs: obs,
            self.model.a: actions,
            self.target_obs: tar_obs,
            self.target_rew: tar_rew,
        }

        ret_vals = [
            self.loss,
            self.obs_loss,
            self.rew_loss,
            self.opt,
        ]

        if summary_op is not None:
            ret_vals.append(summary_op)
        
        return self.sess.run(ret_vals, feed_dict=feed_dict)

    # Given an observation and an action, return the predicted next observation and reward
    def predict(self, obs, a):
        return self.model.predict(obs, a)


def train(env_fn=None,
          spectrum = False,
          em_arch=None,
          a2c_policy=None,
          nenvs = 16,
          nsteps = 100,
          max_iters = 1e6,
          obs_coeff = 0.5,
          rew_coeff = 0.5,
          lr = 7e-4,
          loss='mse',
          log_interval = 100,
          summarize=True,
          em_load_path=None,
          a2c_load_path=None,
          log_path=None,
          cpu_cores=1):

    # Construct the vectorized parallel environments
    envs = [ env_fn for _ in range(nenvs) ]
    envs = SubprocVecEnv(envs)

    # Set some random seeds for the environment
    envs.seed(0)
    if spectrum:
        envs.spectrum()

    ob_space = envs.observation_space.shape
    nw, nh, nc = ob_space
    ac_space = envs.action_space

    obs = envs.reset()
    
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=cpu_cores,
        intra_op_parallelism_threads=cpu_cores )
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:

        actor_critic = RandomActorCritic(sess, a2c_policy, ob_space, ac_space, nenvs, nsteps)

        if a2c_load_path is not None:
            actor_critic.load(a2c_load_path)
            with open(logpath+'/a2c_load_path', 'w') as outfile:
                outfile.write(a2c_load_path)
            print('Loaded a2c')
        else:
            actor_critic.epsilon = -1
            print('WARNING: No Actor Critic Model loaded. Using Random Agent')

        env_model = EnvironmentModel(sess, em_arch, ob_space, ac_space, loss,
                                     lr, obs_coeff, rew_coeff, summarize)

        load_count = 0
        if em_load_path is not None:
            env_model.load(em_load_path)

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path, graph=sess.graph)

        sess.run(tf.global_variables_initializer())

        print('Env Model Training Start!')
        print('Model will saved on intervals of %i' % (log_interval))
        for i in tqdm(range(load_count + 1, int(max_iters)+1), ascii=True, desc='EnvironmentModel'):

            mb_s, mb_a, mb_r, mb_ns, mb_d = [], [], [], [], []
            
            for s, a, r, ns, d in model_play_games(actor_critic, envs, nsteps):
                mb_s.append(s)
                mb_a.append(a)
                mb_r.append(r)
                mb_ns.append(ns)
                mb_d.append(d)

            mb_s = np.concatenate(mb_s)
            mb_a = np.concatenate(mb_a)
            mb_r = np.concatenate(mb_r)
            mb_ns= np.concatenate(mb_ns)
            mb_d = np.concatenate(mb_d)
           
            if summarize:
                loss, obs_loss, rew_loss, _, smy = env_model.train(mb_s, mb_a, mb_ns, mb_r, summary_op)
                writer.add_summary(smy, i)
            else:
                loss, obs_loss, rew_loss, _ = env_model.train(mb_s, mb_a, mb_ns, mb_r)

            if i % log_interval == 0:
                env_model.save(log_path, i)

        env_model.save(log_path, 'final')
        print('Environment Model is finished training')
