#!/usr/bin/env python

import sys
import argparse
import os



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Primary Functionality
    parser.add_argument('--a2c',
            help='Train the Actor-Critic Agent',
            action="store_true")
    parser.add_argument('--em',
            help='Train the Environment Model',
            action="store_true")

    # Secondary Tasks
    parser.add_argument('--a2c-pd-test',
            help='Test the Actor-Critic Params on a single env and show policy logits',
            action="store_true")

    # Environment Arguments
    parser.add_argument('--env',
            help='Environment ID',
            default='cube-x2-v0')
    parser.add_argument('--scramble',
            help='Set the max scramble size. format: size (or) initial:target:episodes',
            type=str, default='1')
    parser.add_argument('--maxsteps',
            help='Set the max step size. format: size (or) initial:target:episodes',
            type=str, default='1')
    parser.add_argument('--adaptive',
            help='Turn on the adaptive curriculum',
            action="store_true")
    parser.add_argument('--spectrum',
            help='Setup up a spectrum of environments with different difficulties',
            action="store_true")
    parser.add_argument('--easy',
            help='Make the environment extremely easy; No orientation change, only R scrabmle',
            action="store_true")
    parser.add_argument('--orient-scramble',
            help='Lets the environment scramble orientation as well',
            action="store_true")

    # Model Free Policy Parameteres
    parser.add_argument('--policy',
            help='Specify the type of policy for the model free part [cnn, mlp]',
            type=str, default='cnn')
    parser.add_argument('--coordConv',
            help='Use the special Coordinate Convolutional Layers',
            action="store_true")

    # Actor Critic Arguments
    parser.add_argument('--workers',
            help='Set the number of workers',
            type=int, default=16)
    parser.add_argument('--nsteps',
            help='Number of environment steps per training iteration per worker',
            type=int, default=40)
    parser.add_argument('--iters',
            help='Number of training iterations',
            type=float, default=5e4)
    parser.add_argument('--a2c-load',
            help='Load Path for the Actor-Critic Parameters',
            type=str, default=None)

    # Other misc arguments
    parser.add_argument('--log-interval',
            help='Set the logging interval',
            type=int, default=100)

    args = parser.parse_args()

    # Verify that atleast one of the primary functions are chosen by the user
    assert sum([args.a2c, args.em, args.a2c_pd_test]) == 1, ""

    if ':' not in args.scramble:
        args.scramble = int(args.scramble)
    if ':' not in args.maxsteps:
        args.maxsteps = int(args.maxsteps)

    # Decide whether we should use the GPU or not...
    # Certain modes should use the GPU, waste of memory
    if args.a2c_pd_test:
        os.environ['CUDA_VISIBLE_DEVICES']="-1"

    # We import the main stuff here, otherwise its really slow
    import gym
    import a2c
    import cube_gym
    from policy import Policies

    def cube_env():
        env = gym.make(args.env)
        env.unwrapped._refresh(args.scramble, args.maxsteps, args.easy, args.adaptive,
                              args.orient_scramble)
        return env

    if args.a2c:
        a2c.train(  env_fn          = cube_env,
                    spectrum        = args.spectrum,
                    policy          = Policies[args.policy],
                    nenvs           = args.workers,
                    nsteps          = args.nsteps,
                    max_iterations  = int(args.iters),
                    gamma           = 0.99,
                    vf_coeff        = 0.5,
                    ent_coeff       = 0.01,
                    max_grad_norm   = 0.5,
                    lr              = 7e-4,
                    alpha           = 0.99,
                    epsilon         = 1e-5,
                    log_interval    = args.log_interval,
                    save_interval   = 1e3,
                    load_count      = 0,
                    summarize       = True,
                    load_path       = args.a2c_load,
                    save_path       = './experiments/a2c/weights',
                    log_path        = './experiments/a2c/logs')
    
    if args.a2c_pd_test:
        a2c.pd_test(env_fn          = cube_env,
                    policy          = Policies[args.policy],
                    load_path       = args.a2c_load)


if __name__ == '__main__':
    main()
