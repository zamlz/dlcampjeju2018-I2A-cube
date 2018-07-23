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
    parser.add_argument('--a2c-pd-test',
            help='Test the Actor-Critic Params on a single env and show policy logits',
            action="store_true")
    parser.add_argument('--em',
            help='Train the Environment Model',
            action="store_true")

    # General Arguments
    parser.add_argument('--iters',
            help='Number of training iterations',
            type=float, default=5e4)


    # Environment Arguments
    parser.add_argument('--env',
            help='Environment ID',
            default='cube-x2-v0')
    parser.add_argument('--workers',
            help='Set the number of workers',
            type=int, default=16)
    parser.add_argument('--nsteps',
            help='Number of environment steps per training iteration per worker',
            type=int, default=40)
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

    # Actor Critic Arguments
    parser.add_argument('--a2c-policy',
            help='Specify the policy architecture',
            type=str, default='c2d+:16:3:1_h:4096:2048_pi_vf')
    parser.add_argument('--a2c-policy-help',
            help='Show the help dialouge to generate a policy string',
            action="store_true")
    parser.add_argument('--a2c-load',
            help='Load Path for the Actor-Critic Weights',
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
            type=float, default=0.05)

    # Environment Model Arguments
    parser.add_argument('--em-arch',
            help='Specify the environment model architecture',
            type=str, default='h:4096:4096:4096')
    parser.add_argument('--em-arch-help',
            help='Show the help dialouge to generate a em arch string',
            action="store_true")
    parser.add_argument('--em-load',
            help='Load Path for the Environment-Model Weights',
            type=str, default=None)
    parser.add_argument('--em-loss',
            help='Specify the loss function for training the Env Model [mse,ent]',
            type=str, default='mse')
    parser.add_argument('--obs-coeff',
            help='Specify the Predicted Observation Loss Coefficient',
            type=float, default=1.0)
    parser.add_argument('--rew-coeff',
            help='Specify the Predicted Reward Loss Coefficient',
            type=float, default=1.0)

    # Other misc arguments
    parser.add_argument('--exp-root',
            help='Set the root path for all experiments',
            type=str, default='./experiments/')
    parser.add_argument('--exppath',
            help='Return the experiment folder under the specified arguments',
            action="store_true")
    parser.add_argument('--tag',
            help='Tag the current experiemnt',
            type=str, default='')
    parser.add_argument('--log-interval',
            help='Set the logging interval',
            type=int, default=1e3)
    parser.add_argument('--cpu',
            help='Set the number of cpu cores available',
            type=int, default=16)
    parser.add_argument('--no-override',
            help='Prevent loading arguments to override default settings',
            action="store_true")

    # Decode User Arguments
    #######################################################################################
    
    args = parser.parse_args()
    if args.a2c_policy_help:
        from actor_critic import PolicyBuilder
        print(PolicyBuilder.__doc__)
        exit(0)

    if args.em_arch_help:
        from environment_model import EMBuilder
        print(EMBuilder.__doc__)
        exit(0)

    # Verify that atleast one of the primary functions are chosen by the user
    assert sum([args.a2c, args.em, args.a2c_pd_test]) == 1, ""

    if ':' not in args.scramble:
        args.scramble = int(args.scramble)
    if ':' not in args.maxsteps:
        args.maxsteps = int(args.maxsteps)

    # Create the logging path
    logpath = args.exp_root 

    if args.a2c:
        logpath += 'a2c/'
    elif args.em:
        logpath += 'em/'

    if args.a2c:
        logpath += args.a2c_policy + '/'
    if args.em:
        logpath += args.em_arch + '/'
    else:
        logpath += 'NULL/'

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
    
    if args.a2c:
        logpath += 'pgk_' + str(args.pg_coeff) + '/'
        logpath += 'vfk_' + str(args.vf_coeff) + '/'
        logpath += 'entk_' + str(args.ent_coeff) + '/'
  
    if args.em:
        logpath += 'obk_' + str(args.obs_coeff) + '/'
        logpath += 'rwk_' + str(args.rew_coeff) + '/'

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
    # to view the help dialouges
    import gym
    import cube_gym
    import actor_critic as a2c
    import environment_model as em

    # Store the default values of some arguments
    scramble        = args.scramble
    maxsteps        = args.maxsteps
    easy            = args.easy
    adaptive        = args.adaptive
    spectrum        = args.spectrum
    orient_scramble = not args.no_orient_scramble
    a2c_policy_def = args.a2c_policy
    em_arch_def = args.em_arch

    # Override defaults for A2C if user decides to load weights from filesystem
    if (args.a2c or args.a2c_pd_test) and args.a2c_load and not args.no_override:
        
        a2c_load_list = args.a2c_load.split('/')
        adaptive = False
        easy = False
        orient_scramble = False

        for acpd in a2c_load_list:
            if '_pi' in acpd:
                a2c_policy_def = acpd
            if 'adaptive' in acpd:
                adaptive = True
            if 'spectrum' in acpd:
                spectrum = True
            if 'os_yes' in acpd:
                orient_scramble = True
            if 'easy' in acpd:
                easy = True

    # If we are doing a2c pd testing, then we want to use the scramble and maxsteps
    # passed via the command line and not the trained environment defaults
    if args.a2c_pd_test:
        adaptive = False
        spectrum = False
        easy = False
        scramble = args.scramble
        maxsteps = args.maxsteps

    # Override defaults for Environment Model if user decides to load weights form FS
    if args.em and args.em_load and not args.no_override:
        
        em_load_list = args.em_load.split('/')
        adaptive = False
        easy = False
        orient_scramble = False

        for empd in em_load_list:
            if 'adaptive' in empd:
                adaptive = True
            if 'spectrum' in empd:
                spectrum = True
            if 'os_yes' is empd:
                orient_scramble = True
            if 'easy' is acpd:
                easy = True

    # A helper function that returns the correct environment as specified with our
    # settings chosen by the user
    def cube_env():
        env = gym.make(args.env)
        env.unwrapped._refresh(scramble, maxsteps, easy, adaptive, orient_scramble)
        return env

    # Actor Critic Related Stuff
    #######################################################################################

    if args.a2c:
        a2c.train(  env_fn          = cube_env,
                    spectrum        = spectrum,
                    policy          = a2c_policy_def,
                    nenvs           = args.workers,
                    nsteps          = args.nsteps,
                    max_iters       = int(args.iters),
                    gamma           = 0.99,
                    pg_coeff        = args.pg_coeff, 
                    vf_coeff        = args.vf_coeff,
                    ent_coeff       = args.ent_coeff,
                    max_grad_norm   = 0.5,
                    lr              = args.lr,
                    alpha           = 0.99,
                    epsilon         = 1e-5,
                    log_interval    = args.log_interval,
                    summarize       = True,
                    load_path       = args.a2c_load,
                    log_path        = logpath, 
                    cpu_cores       = args.cpu)
    
    if args.a2c_pd_test:
        a2c.pd_test(env_fn          = cube_env,
                    policy          = a2c_policy_def,
                    load_path       = args.a2c_load)

    # Environment Model Related Stuff
    #######################################################################################

    if args.em:
        em.train(   env_fn          = cube_env,
                    spectrum        = spectrum,
                    em_arch         = em_arch_def,
                    a2c_policy      = a2c_policy_def,
                    nenvs           = args.workers,
                    nsteps          = args.nsteps,
                    max_iters       = int(args.iters),
                    obs_coeff       = args.obs_coeff,
                    rew_coeff       = args.rew_coeff,
                    lr              = args.lr,
                    loss            = args.em_loss,
                    log_interval    = args.log_interval,
                    summarize       = True,
                    em_load_path    = args.a2c_load,
                    a2c_load_path   = args.a2c_load,
                    log_path        = logpath, 
                    cpu_cores       = args.cpu)
    


if __name__ == '__main__':
    main()
