
import gym
import cube_gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env = gym.make('cube-x3-v0')
env.unwrapped._refreshScrambleParameters(1, 100)
obs = env.reset()

action_list = [ env.action_space.sample() for _ in range(50) ]

fig = plt.figure()
ims = []

for a in action_list:
    obs, r, d, _ = env.step(a)

    im = plt.imshow(cube_gym.onehotToRGB(obs))
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=2000, blit=True, repeat_delay=4000)

plt.show()
