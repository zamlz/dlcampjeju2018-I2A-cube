Imagination-Augmented Agents for Deep Reinforcement Learning to Solve Rubik's Cubes
===================================================================================
![Random Exploration Cube][cube-gif]
Jeju Deep Learning Camp 2018 [ Amlesh Sivanantham ]
---------------------------------------------------

To solve a Rubik's Cube environment with the model prescribed in the paper,
[Imagination-Augmented Agents for Deep Reinforcement Learning, *Weber et al*][i2a-paper].
The I2A model generates observation predictions from an learned environment model.
I2A's learn to leverage multiple rollouts of these predicitions to construct a
better policy and value network for the agent. 



[i2a-paper]: https://arxiv.org/abs/1707.06203v2

[cube-gif]: https://raw.githubusercontent.com/zamlz/dlcampjeju2018-I2A-cube/master/docs/pics/cube_solve.gif
