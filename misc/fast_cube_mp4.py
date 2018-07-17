
import gym
import sys
import cube_gym
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env = gym.make('cube-x3-v0')
env.unwrapped._refresh(26, 30)
actions = env.unwrapped.action_list

obs = env.reset()
env.render()

actions = env.unwrapped.scramble_current[::-1]
actions = [ env.unwrapped.action_list.index(x) for x in actions ]
actions = [ x+6 if x < 6 else x % 6 for x in actions ]

fig = plt.figure()
ims = []
writer = animation.writers['ffmpeg']
writer = writer(fps=5, metadata=dict(artist='zamlz'),bitrate=18000)

im = plt.imshow(cube_gym.onehotToRGB(obs))
ims.append([im])


for a in actions: 
    obs, r, d, _ = env.step(a)

    obs = cube_gym.onehotToRGB(obs)
    im = plt.imshow(obs)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=2000, blit=True, repeat_delay=2000)
# plt.show()
ani.save('cubeSolve.mp4',writer=writer)
