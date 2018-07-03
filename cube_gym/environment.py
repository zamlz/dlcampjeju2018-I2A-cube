
#                 _                                      _                 
#  ___ _ ____   _(_)_ __ ___  _ __  _ __ ___   ___ _ __ | |_   _ __  _   _ 
# / _ \ '_ \ \ / / | '__/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __| | '_ \| | | |
#|  __/ | | \ V /| | | | (_) | | | | | | | | |  __/ | | | |_ _| |_) | |_| |
# \___|_| |_|\_/ |_|_|  \___/|_| |_|_| |_| |_|\___|_| |_|\__(_) .__/ \__, |
#                                                             |_|    |___/ 

#
#   The Gym environment wrapper for the cube.py program.
#   The cube.py program is a standalone cli application
#   for solving rubik's cubes of any size with a variety
#   of features. This file aims to add the necessary wrappings
#   to make it a suitable gym environment to be used with
#   any RL algorithm that uses gym.

import gym
import random
import numpy as np
import gym.spaces as spaces
import cube_gym.cube as cg

from gym.utils import seeding


class CubeEnv(gym.Env):
    """
    Generate the Cube Gym Environment under the following arguments.
        order           - Specify the size of the cube environment
        reward_type     - Specify which reward function to train with
        scramble_depth  - Specify a static depth of scrambling to use, if
                          it is None, then we must recover a scramble depth
                          from the reset command.
        max_steps       - The maximum number of attemps allowed to take
                          before the environment resets
    """

    def __init__(self, order, reward_type='sparse', scramble_type='static',
                 scramble_depth=4, max_steps=10):

        self.order = order
        # Actions spaces 3 and under, only have 12 face moves (middle turns can be thought of
        # as functions of the other moves) and 6 orientation moves.
        if order <= 3:
            self.action_list = [
            # Clockwise face turns
            'r','l','u','d','f','b',
            # Counter-Clockwise face turns
            '.r','.l','.u','.d','.f','.b',
            # Orientation Manipulation
            'x', 'y', 'z', '.x', '.y', '.z',
            ]
            self.action_space = spaces.Discrete(len(self.action_list))
        else:
            raise NotImplemented('Generation of Action Space past order 3 is not implemented')

        # The image of an unpacked cube is fit to the pixel. That means that the maximum
        # width of the cube is 4 times the order size and we match the height of the cube
        # to the criteria as well. It contains 6 channels, one for each color type.
        imglen = order*4
        self.observation_space = spaces.Box(-1, 1, (imglen, imglen, 6), dtype=np.float32)

        # Choose the specified reward function
        reward_funcs = {
            'sparse': self._sparse_reward,
        }
        self.reward_function = reward_funcs[reward_type]

        self.scramble_depth = scramble_depth
        self.steps = 0
        self.cube = cg.Cube(order=order)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # -------------------- Reward Functions ---------------------------------
    # all reward functions should specify if the environment is done or not

    # Generates a positive reward only when the cube is solved.
    def _sparse_reward(self):
        if self.cube.isSolved():
            return 1.0, True
        return 0.0, False

    # -----------------------------------------------------------------------

    # Step the environment
    def step(self, action):
        self.steps += 1
        self.cube.minimalInterpreter(self.action_list[action])
        
        reward, rdone = self.reward_function()
        done = (self.steps > self.max_steps) or rdone
        
        img = self._genImgStateOneHot()

        return img, reward, done, {}

    # Reset the environment
    def reset(self, *args, **kwargs):
        self.cube.restoreSolvedState()

        if self.scramble_depth:
            scramble = self.scramble_depth

        # Perform some env scrambling here
        while self.cube.isSolved():
            # Keep attempting scrambles until we don't end up in a solved state
            for _ in range(scramble):
                self.cube.minimalInterpreter(self.action_list[self.action_space.sample()])

        return self._genImgStateOneHot()

    # Render the environment
    def render(self):
        self.cube.displayCube(isColor=True)

    # Generate the image state in the one hot format (6 Channels)
    def _genImgStateOneHot(self):
        state = np.zeros(self.observation_space.shape)
        # Unfortunetly, we need to loop through every face to get construct this matrix

        # Up
        for i in range(self.order):
            for j in range(self.order):
                state[i][j + self.order] = cg.tileDictOneHot[self.cube.up[i][j]]

        # Left
        for i in range(self.order):
            for j in range(self.order):
                state[i + self.order][j] = cg.tileDictOneHot[self.cube.left[i][j]]
        
        # Front
        for i in range(self.order):
            for j in range(self.order):
                state[i + self.order][j + self.order] = cg.tileDictOneHot[self.cube.front[i][j]]
        
        # Right
        for i in range(self.order):
            for j in range(self.order):
                state[i + self.order][j + 2*self.order] = cg.tileDictOneHot[self.cube.right[i][j]]
        
        # Back
        for i in range(self.order):
            for j in range(self.order):
                state[i + self.order][j + 3*self.order] = cg.tileDictOneHot[self.cube.back[i][j]]
        
        # Down
        for i in range(self.order):
            for j in range(self.order):
                state[i + 2*self.order][j + self.order] = cg.tileDictOneHot[self.cube.down[i][j]]

        return state

