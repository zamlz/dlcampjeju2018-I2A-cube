
import gym
import numpy as np
import tensorflow as tf
import time

from actor_critic.policy import A2CBuilder 
from actor_critic.util import discount_with_dones, cat_entropy, fix_tf_name
from common.model import NetworkBase
from common.multiprocessing_env import SubprocVecEnv
from tqdm import tqdm


class ActorCritic(NetworkBase):

    def __init__(self, sess, a2c_arch, ob_space, ac_space, nenvs, nsteps,
                 pg_coeff=1.0, vf_coeff=0.5, ent_coeff=0.01, max_grad_norm=0.5,
                 lr=7e-4, alpha=0.99, epsilon=1e-5, summarize=False):

        self.sess = sess
        self.nact = ac_space.n
        nbatch = nenvs * nsteps

        # Actions, Advantages, and Reward
        self.actions = tf.placeholder(tf.int32, [nbatch], name='actions')
        self.advantages = tf.placeholder(tf.float32, [nbatch], name='advantages')
        self.rewards = tf.placeholder(tf.float32, [nbatch], name='rewards')
        self.depth = tf.placeholder(tf.float32, [nbatch], name='scramble_depth')
      
        # setup the models
        self.step_model = A2CBuilder(self.sess, ob_space, ac_space, nenvs, 1,
                                        reuse=False, build=a2c_arch)
        self.train_model = A2CBuilder(self.sess, ob_space, ac_space, nbatch, nsteps,
                                         reuse=True, build=a2c_arch)

        # Negative log probs of actions
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.train_model.pi, labels=self.actions)

        # Policy Gradients Loss, Value Function Loss, Entropy, and Full Loss
        self.pg_loss = tf.reduce_mean(self.advantages * neglogpac)
        self.vf_loss = tf.reduce_mean(tf.square(tf.squeeze(self.train_model.vf) - self.rewards) / 2.0)
        self.entropy = tf.reduce_mean(cat_entropy(self.train_model.pi))
        self.loss = pg_coeff*self.pg_loss - ent_coeff*self.entropy + vf_coeff*self.vf_loss
        
        self.mean_rew= tf.reduce_mean(self.rewards)
        self.mean_depth = tf.reduce_mean(self.depth)

        # Find the model parameters and their gradients
        with tf.variable_scope('a2c_model'):
            self.params = tf.trainable_variables()
        grads = tf.gradients(self.loss, self.params)

        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, self.params))

        # Setup the optimizer
        trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
        self.opt = trainer.apply_gradients(grads)

        # For some awesome tensorboard stuff
        if summarize:
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Entropy', self.entropy)
            tf.summary.scalar('Policy Gradient Loss', self.pg_loss)
            tf.summary.scalar('Value Function Loss', self.vf_loss)
            tf.summary.scalar('Rewards', self.mean_rew)
            tf.summary.scalar('Depth', self.mean_depth)

        # fix tf scopes if we are loading a scope that is different from the saved instance
        #name_scope = tf.contrib.framework.get_name_scope()
        #if len(name_scope) != 0:
        #    self.params = { fix_tf_name(v.name, name_scope): v for v in self.params }
        #else:
        #    self.params = { fix_tf_name(v.name): v for v in self.params }

        # Initialize the tensorflow saver
        self.saver = tf.train.Saver(self.params, max_to_keep=5)

    # Single training step
    def train(self, obs, rewards, masks, actions, values, depth, step, summary_op=None):
        advantages = rewards - values

        feed_dict = {
            self.actions: actions,
            self.advantages: advantages,
            self.rewards: rewards,
            self.depth: depth,
        }

        inputs = self.train_model.get_inputs()
        mapped_input = self.train_model.transform_input(obs)
        for transformed_input, inp in zip(mapped_input, inputs):
            feed_dict[inp] = transformed_input

        ret_vals = [
            self.loss,
            self.pg_loss,
            self.vf_loss,
            self.entropy,
            self.mean_rew,
            self.mean_depth,
            self.opt,
        ]

        if summary_op is not None:
            ret_vals.append(summary_op)

        return self.sess.run(ret_vals, feed_dict=feed_dict)

    # Given an observation, perform an action
    def act(self, obs, stochastic=True):
        return self.step_model.step(obs, stochastic=stochastic)

    # Return the value of the value function
    def critique(self, obs):
        return self.step_model.value(obs)

# The function that trains the a2c model

def train(env_fn        = None,
          spectrum      = False,
          a2c_arch      = None,
          nenvs         = 16,
          nsteps        = 100,
          max_iters     = 1e6,
          gamma         = 0.99,
          pg_coeff      = 1.0,
          vf_coeff      = 0.5,
          ent_coeff     = 0.01,
          max_grad_norm = 0.5,
          lr            = 7e-4,
          alpha         = 0.99,
          epsilon       = 1e-5,
          log_interval  = 100,
          summarize     = True,
          load_path     = None,
          log_path      = None,
          cpu_cores     = 1):

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

        actor_critic = ActorCritic(sess, a2c_arch, ob_space, ac_space, nenvs, nsteps,
                                   pg_coeff, vf_coeff, ent_coeff, max_grad_norm,
                                   lr, alpha, epsilon, summarize)

        load_count = 0
        if load_path is not None:
            actor_critic.load(load_path)
            print('Loaded a2c')

        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path, graph=sess.graph)

        sess.run(tf.global_variables_initializer())

        batch_ob_shape = (nenvs*nsteps, nw, nh, nc)

        dones = [False for _ in range(nenvs)]
        nbatch = nenvs * nsteps

        episode_rewards = np.zeros((nenvs, ))
        final_rewards = np.zeros((nenvs, ))

        print('a2c Training Start!')
        print('Model will be saved on intervals of %i' % (log_interval))
        for i in tqdm(range(load_count + 1, int(max_iters) + 1), ascii=True, desc='ActorCritic'):
           
            # Create the minibatch lists
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_depth = [], [], [], [], [], []
            total_reward = 0

            for n in range(nsteps):
               
                # Get the actions and values from the actor critic, we don't need neglogp
                actions, values, neglogp = actor_critic.act(obs)
               
                mb_obs.append(np.copy(obs))
                mb_actions.append(actions)
                mb_values.append(values)
                mb_dones.append(dones)

                obs, rewards, dones, info = envs.step(actions)
                total_reward += np.sum(rewards)

                episode_rewards += rewards
                masks = 1 - np.array(dones)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                mb_rewards.append(rewards)
                mb_depth.append(np.array([ info_item['scramble_depth'] for info_item in info ]))

            mb_dones.append(dones)

            # Convert batch steps to batch rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1,0).reshape(batch_ob_shape)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1,0)
            mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1,0)
            mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1,0)
            mb_dones = np.asarray(mb_dones, dtype=np.float32).swapaxes(1,0)
            mb_depth = np.asarray(mb_depth, dtype=np.int32).swapaxes(1,0)
            mb_masks = mb_dones[:, :-1]
            mb_dones = mb_dones[:, 1:]

            last_values = actor_critic.critique(obs).tolist()

            # discounting
            for n, (rewards, d, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                d = d.tolist()
                if d[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], d+[0], gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, d, gamma)
                mb_rewards[n] = rewards

            # Flatten the whole minibatch
            mb_rewards = mb_rewards.flatten()
            mb_actions = mb_actions.flatten()
            mb_values = mb_values.flatten()
            mb_masks = mb_masks.flatten()
            mb_depth = mb_depth.flatten()

            # Save the information to tensorboard
            if summarize:
                loss, policy_loss, value_loss, policy_ent, mrew, mdp, _, summary = actor_critic.train(
                        mb_obs, mb_rewards, mb_masks, mb_actions, mb_values, mb_depth, i, summary_op)
                writer.add_summary(summary, i)
            else:
                loss, policy_loss, value_loss, policy_ent, mrew, mdp, _ = actor_critic.train(
                        mb_obs, mb_rewards, mb_masks, mb_actions, mb_values, mb_depth, i)

            if i % log_interval == 0:
                actor_critic.save(log_path, i)
            
        actor_critic.save(log_path, 'final')
        print('a2c model is finished training')


