
import gym
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from vautoencoder.network import VAEBuilder
from actor_critic import RandomActorCritic
from common.multiprocessing_env import SubprocVecEnv
from common.model import NetworkBase, model_play_games

class VariationalAutoEncoder(NetworkBase):
    
    def __init__(self, sess, var_arch, ob_space, ac_space, lr=0.001, kl_coeff=0.5, summarize=False):

        self.sess = sess
        self.nact = ac_space.n
        nw, nh, nc = ob_space

        # Setup targets
        self.target_obs = tf.placeholder(tf.float32, [None, nw, nh, nc], name='target_obs')

        # Setup the network
        self.vae = VAEBuilder(sess, var_arch, ob_space, ac_space)

        # Compute losses
        logits_flat = tf.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.target_obs)

        self.reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis=1)
        self.kl_loss = tf.exp(self.var.z_logvar) + self.vae.z_mu**2 - 1 - self.var.z_logvar
        self.kl_loss = kl_coeff*tf.reduce_sum(self.kl_loss, 1)
        self.loss = tf.reduce_mean(self.reconstruction_loss + self.kl_loss)

        # Find the model parameters
        with tf.variable_scope('vae_model'):
            self.params = tf.trainable_variables()
        grads = tf.gradients(self.loss, self.params)
        grads = list(zip(grads, self.params))

        # Setup the optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=lr)
        self.opt = trainer.apply_gradients(grads)

        if summarize:
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('KL Loss', self.kl_loss)
            tf.summary.scalar('Reconstruction Loss', self.reconstruction_loss)

        self.saver = tf.train.Saver(self.params, max_to_keep=5)


    def train(self, obs, actions, tar_obs, tar_rew, summary_op=None):
        
        feed_dict = {
            self.vae.obs: obs,
            self.var.a: actions,
            self.target_obs: tar_obs,
            self.target_rew: tar_rew,
        }

        ret_vals = [
            self.loss,
            self.reconstruction_loss,
            self.kl_loss,
            self.opt,
        ]

        if summary_op is not None:
            ret_vals.append(summary_op)

        return self.sess.run(ret_vals, feed_dict=feed_dict)


def train(env_fn=None,
          spectrum=False,
          vae_arch=None,
          a2c_policy=None,
          nenvs=16,
          nsteps=100,
          max_iters=1e6,
          kl_coeff=0.5,
          lr = 7e-4,
          log_interval=100,
          summarize=True,
          vae_load_path=None,
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
    tf_config.gpu_options.allow_growth =True

    with tf.Session(config=tf_config) as sess:

        actor_critic = RandomActorCritic(sess, a2c_policy, ob_space, ac_space, nenvs, nsteps)

        if a2c_load_path is not None:
            actor_critic.load(a2c_load_path)
            print('Loaded a2c')
        else:
            actor_critic.epsilon = -1
            print('WARNING: No Actor Critic Model loaded. Using Random Agent')

        vae = VariationalAutoEncoder(sess, vae_arch, ob_space, ac_space, lr, kl_coeff, summarize) 

        load_count = 0
        if vae_load_path is not None:
            vae.load(vae_load_path)

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path, graph=sess.graph)

        sess.run(tf.global_variables_initializer())

        print('VAE Training Start!')
        print('Model will be saved on intervals of %i' % (log_interval))
        for i in tqdm(range(load_count + 1, int(max_iters)+1), ascii=True, desc='VarAutoEncoder'):

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
                loss, recon_loss, kl_loss, _, smy = vae.train(mb_s, mb_a, mb_ns, mb_r, summary_op) 
                writer.add_summary(smy, i)
            else:
                loss, recon_loss, kl_loss, _ = vae.train(mb_s, mb_a, mb_ns, mb_r)

            if i % log_interval == 0:
                vae.save(log_path, i)

        vae.save(log_path, 'final')
        print('Variational AutoEncoder is finished training')
