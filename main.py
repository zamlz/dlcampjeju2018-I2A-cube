#!/usr/bin/env python

import sys
import argparse
import datetime
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
    parser.add_argument('--tag',
            help='Tag the current experiemnt',
            type=str, default='')

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
    parser.add_argument('--no-orient-scramble',
            help='Lets the environment scramble orientation as well',
            action="store_true")

    # Model Free Policy Parameteres
    parser.add_argument('--policy',
            help='Specify the policy architecture',
            type=str, default='c2d+:16:3:1_h:4096:2048_pi_vf')
    parser.add_argument('--policy-help',
            help='Show the help dialouge to generate a policy string',
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
    parser.add_argument('--lr',
            help='Specify the learning rate to use',
            type=float, default=7e-4)
    parser.add_argument('--pg-coeff',
            help='Specify the Policy Gradient Loss Coefficient',
            type=float, default=1.0)
    parser.add_argument('--vf-coeff',
            help='Specify the Value Function Loss Coefficient',
            type=float, default=0.5)
    parser.add_argument('--ent-coeff',
            help='Specify the Entropy Coefficient',
            type=float, default=0.01)

    # Other misc arguments
    parser.add_argument('--log-interval',
            help='Set the logging interval',
            type=int, default=100)
    parser.add_argument('--cpu',
            help='Set the number of cpu cores available',
            type=int, default=16)
    parser.add_argument('--exppath',
            help='Return the experiment folder under the specified arguments',
            action="store_true")

    args = parser.parse_args()
    if args.policy_help:
        from policy import PolicyBuilder
        print(PolicyBuilder.__doc__)
        exit(0)

    # Verify that atleast one of the primary functions are chosen by the user
    assert sum([args.a2c, args.em, args.a2c_pd_test]) == 1, ""

    if ':' not in args.scramble:
        args.scramble = int(args.scramble)
    if ':' not in args.maxsteps:
        args.maxsteps = int(args.maxsteps)

    # Create the logging paths
    logpath = './experiments/'

    if args.a2c:
        logpath += 'a2c/'
    elif args.em:
        logpath += 'em/'

    logpath += args.policy + '/'

    logpath += args.env + '/'

    if args.adaptive:
        logpath += 'adaptive/'
    elif args.spectrum:
        logpath += 'spectrum/'
    elif args.easy:
        logpath += 'easy/'
    else:
        logpath += 's_' + str(args.scramble) + '_m_' + str(args.maxsteps) + '/'

    if args.no_orient_scramble:
        logpath += 'os_no/'
    else:
        logpath += 'os_yes/'

    logpath += 'iter_' + str(args.iters) + '/'
    logpath += 'lr_' + str(args.lr) + '/'
    logpath += 'pgk_' + str(args.pg_coeff) + '/'
    logpath += 'vfk_' + str(args.vf_coeff) + '/'
    logpath += 'entk_' + str(args.ent_coeff) + '/'
   
    if args.tag == '':
        if args.exppath:
            print(logpath)
            exit()
        else:
            logpath += datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f/")
    else:
        logpath += args.tag + '/'
        if args.exppath:
            print(logpath)
            exit()

    # Decide whether we should use the GPU or not...
    # Certain modes should use the GPU, waste of memory
    if args.a2c_pd_test:
        os.environ['CUDA_VISIBLE_DEVICES']="-1"

    # We import the main stuff here, otherwise its really slow
    import gym
    import a2c
    import cube_gym
    from policy import PolicyBuilder, policy_parser

    def cube_env():
        env = gym.make(args.env)
        env.unwrapped._refresh(args.scramble, args.maxsteps, args.easy, args.adaptive,
                              not args.no_orient_scramble)
        return env

    
    builder = policy_parser(args.policy)
    def policy_fn(sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        pi = PolicyBuilder(sess=sess, ob_space=ob_space, ac_space=ac_space,
                           nbatch=nbatch, nsteps=nsteps, reuse=reuse,
                           build=builder)
        return pi
        

    if args.a2c:
        a2c.train(  env_fn          = cube_env,
                    spectrum        = args.spectrum,
                    policy          = policy_fn,
                    nenvs           = args.workers,
                    nsteps          = args.nsteps,
                    max_iterations  = int(args.iters),
                    gamma           = 0.99,
                    pg_coeff        = args.pg_coeff, 
                    vf_coeff        = args.vf_coeff,
                    ent_coeff       = args.ent_coeff,
                    max_grad_norm   = 0.5,
                    lr              = args.lr,
                    alpha           = 0.99,
                    epsilon         = 1e-5,
                    log_interval    = args.log_interval,
                    save_interval   = 1e3,
                    load_count      = 0,
                    summarize       = True,
                    load_path       = args.a2c_load,
                    save_path       = logpath,
                    log_path        = logpath, 
                    cpu_cores       = args.cpu)
    
    if args.a2c_pd_test:
        a2c.pd_test(env_fn          = cube_env,
                    policy          = Policies[args.policy],
                    load_path       = args.a2c_load)


if __name__ == '__main__':
    main()
