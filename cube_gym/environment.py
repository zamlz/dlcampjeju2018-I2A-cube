
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

    def __init__(self, order, reward_type='sparse', scramble_depth=1, max_steps=3, noise=0.0,
                 unbound=False):
        
        self.order = order
        self.agent_solved = False
        self.noise = noise
        self.unbound = unbound

        # Actions spaces 3 and under, only have 12 face moves (middle turns can be thought of
        # as functions of the other moves) and 6 orientation moves.
        if order <= 3:

            self.face_action_list = [
                # Clockwise face turns
                'r','l','u','d','f','b',
                # Counter-Clockwise face turns
                '.r','.l','.u','.d','.f','.b',
            ]

            self.orient_action_list = [
                # Orientation Manipulation
                'x', 'y', 'z', '.x', '.y', '.z',
            ]

        else:
            raise NotImplemented('Generation of Action Space past order 3 is not implemented')

        # The image of an unpacked cube is fit to the pixel. That means that the maximum
        # width of the cube is 4 times the order size and we match the height of the cube
        # to the criteria as well. It contains 6 channels, one for each color type.
        imglen = order*4
        self.observation_space = spaces.Box(-1, 1, (imglen, imglen, 6), dtype=np.float32)
        self.action_space = None # Its set in the refresh command!!!
        self._refresh(scramble_depth, max_steps)

        # Choose the specified reward function
        reward_funcs = {
            'sparse': self._sparse_reward,
            'naive': self._naive_reward,
        }
        self.reward_function = reward_funcs[reward_type]

        self.steps = 0
        self.cube = cg.Cube(order=order)
        self._seed(seed=None)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # You can update the parameters of scrambling and max_steps
    # String inputs are of the form 'initial:final:episodes'
    #
    # Otherwise you can provide a constant input,
    #
    # Why is this a seperate function? So you may control these parameters
    # even after you've initialized the environment
    #
    # Adaptive is another type of curriculum, it update the scramble size if
    # you solve the cube correctly and decreases it if you can't.
    def _refresh(self, scramble_depth='1:4:10', max_steps='10:20:10',
            scramble_easy=False, adaptive=False, orient_scramble=False, noise=0.0):

        self.scramble_easy = scramble_easy
        self.adaptive_curriculum = adaptive
        self.orient_scramble = orient_scramble
        self.noise = noise
       
        # This is updated here because if we decide to orient scramble our action
        # space changes. THEREFORE, you should never change the orient scramble
        # parameter after you started 
        if orient_scramble:
            self.action_list = self.face_action_list + self.orient_action_list
        else:
            self.action_list = self.face_action_list
        self.action_space = spaces.Discrete(len(self.action_list))

        self.scramble_update = 0
        self.max_steps_update = 0

        # Extract the linear scramble curriculum from the string
        if type(scramble_depth) is str: 
            scramlist = [ int(x) for x in scramble_depth.split(':') ]
            self.scramble_update = float((scramlist[1] - scramlist[0]) / scramlist[2])
            scramble_depth = scramlist[0]
      
        # Extract the linear step-time curriculum from the string
        if type(max_steps) is str:
            stepslist = [ int(x) for x in max_steps.split(':') ]
            self.max_steps_update = float((stepslist[1] - stepslist[0]) / stepslist[2])
            max_steps = stepslist[0]

        self.scramble_depth = int(scramble_depth)
        self.max_steps = int(max_steps)
        

    # -------------------- Reward Functions ---------------------------------
    # all reward functions should specify if the environment is done or not

    # Generates a positive reward only when the cube is solved.
    def _sparse_reward(self):
        if self.cube.isSolved():
            self.agent_solved = True
            return 1.0, True
        return 0.0, False

    # Generates a reward based on how many like colored tiles are next to
    # match the center piece.
    def _naive_reward(self):
        c = self.cube.contiguousCount(normalize=True)
        e = 3
        return c**e, False

    # -----------------------------------------------------------------------

    # Step the environment
    def step(self, action):
        self.steps += 1
        self.cube.minimalInterpreter(self.action_list[action])
        
        reward, rdone = self.reward_function()
        done = (self.steps >= int(self.max_steps)) or rdone
        
        img = self._genImgStateOneHot()

        return img, reward, done, {'scramble_depth': self.scramble_depth, 'obs': img}

    # Reset the environment
    def reset(self, *args, **kwargs):
        self.cube.restoreSolvedState()

        # This is the linear scramble update curriculum
        # if its not set via the _refresh funtion
        # then these parameters lines below don't do anyting
        self.scramble_depth += self.scramble_update
        self.max_steps += self.max_steps_update

        if self.unbound:
            self.max_steps = 50
        
        # This is some really really easy testing code
        if self.scramble_easy:
            # Only do a 'r' turn for the scramble. The solution is '.r'
            self.cube.minimalInterpreter(self.face_action_list[0])
            return self._genImgStateOneHot()

        # If you're using the adaptive curriculum
        if self.adaptive_curriculum:
            drate = 1
            mrate = 1
            if self.agent_solved:
                self.scramble_depth = drate + self.scramble_depth
                self.max_steps = mrate + self.max_steps
            elif not self.agent_solved and self.scramble_depth > 1:
                self.scramble_depth = max(1, self.scramble_depth - drate)
                self.max_steps = max(1, self.max_steps - mrate)
        
        self.agent_solved = False
        self.steps = 0
        scramble = int(self.scramble_depth)

        # Perform some env scrambling here
        while self.cube.isSolved():
            # Keep attempting scrambles until we don't end up in a solved state
            scramble_actions = []

            # We first choose a random orientation
            if self.orient_scramble:
                for _ in range(10):
                    scramble_actions.append(self.np_random.choice(self.orient_action_list))
                    self.cube.minimalInterpreter(scramble_actions[-1])
            
            # Now scramble the moves.
            for _ in range(scramble):
                scramble_actions.append(self.np_random.choice(self.face_action_list))
                self.cube.minimalInterpreter(scramble_actions[-1])
        self.scramble_current = scramble_actions

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

        # Add some noise to the observation
        n = self.noise
        # state += n * (np.random.sample(size=state.shape) - 0.5)
        n = min(1.0, n)
        state = np.multiply(state, np.ones(shape=state.shape) - n*np.random.sample(size=state.shape))
    
        return state


# This function is used to convert the 6-channel representation to 3 channel rgb representation
def onehotToRGB(obs, clean=False):
    height, width, _ = obs.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    if clean:
        obs = np.round(obs)

    for i in range(height):
        for j in range(width):
           
            # Red
            if obs[i, j, 0]:
                rgb[i, j] = [1, 0, 0]

            # Orange (Purple)
            elif obs[i, j, 1]:
                rgb[i, j] = [1, 0, 1]

            # Yellow
            elif obs[i, j, 2]:
                rgb[i, j] = [1, 1, 0]

            # Green
            elif obs[i, j, 3]:
                rgb[i, j] = [0, 1, 0]

            # Blue
            elif obs[i, j, 4]:
                rgb[i, j] = [0, 0, 1]

            # White (Cyan)
            elif obs[i, j, 5]:
                rgb[i, j] = [0, 1, 1]

            # Empty
            else:
                rgb[i, j] = [0, 0, 0]

    return rgb

def onehotToRGBNoise(obs):
    height, width, _ = obs.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    for i in range(height):
        for j in range(width):
           
            # Red
            rgb[i, j] += np.array([1, 0, 0]) * obs[i, j, 0]

            # Orange (Purple)
            rgb[i, j] += np.array([1, 0, 1]) * obs[i, j, 1]

            # Yellow
            rgb[i, j] += np.array([1, 1, 0]) * obs[i, j, 2]

            # Green
            rgb[i, j] += np.array([0, 1, 0]) * obs[i, j, 3]

            # Blue
            rgb[i, j] += np.array([0, 0, 1]) * obs[i, j, 4]

            # White (Cyan)
            rgb[i, j] += np.array([0, 1, 1]) * obs[i, j, 5]

    return rgb

   
