
import gym
import a2c
import cube_gym
from policy import *

def cube_env_simple():
    env = gym.make('cube-x2-v0')
    env.unwrapped._refreshScrambleParameters(1, 10, scramble_easy=True)
    return env

a2c.train(env_fn=cube_env_simple,
          policy=CnnPolicy,
          nenvs=4,
          nsteps=100,
          max_iterations=1e4,
          gamma=0.99,
          vf_coeff = 0.5,
          ent_coeef = 0.01,
          max_grad_norm = 0.5,
          lr = 7e-4,
          alpha = 0.99,
          epsilon = 1e-5,
          log_interval=100,
          save_interval=1e3,
          save_name='a2c',
          load_count=0,
          summarize=True,
          load_path=None,
          save_path='./experiments/a2c/weights',
          log_path='./experiments/a2c/logs')
 
