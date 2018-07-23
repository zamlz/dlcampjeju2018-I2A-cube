
import numpy as np
from actor_critic import ActorCritic


# Behaves exactly like the actor critic agent,
# but we can introduce more randomness in its output,
# this *may* make it easier to train the env model
# This is entirely optionaly and can be turned off
# by settings the epsilon value to a number smaller
# than zero.
#
# My reasoning for even having this class is because
# there may be situations where the actor critic agent
# may be veyr likely to only choose one action and no
# other actions (when close to the goal state) when
# the environment model should be more focuses on the
# understanding the dynamics of the environment.
class RandomActorCritic(ActorCritic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.epsilon = 0.25 # This value is a hyperparameter

    def act(self, obs, stochastic=True):
        a, v, n = self.step_model.step(obs, stochastic=stochastic)
        if self.np_random.sample > self.epsilon:
            return a, v, n
        else:
            return np.random.randint(self.nact, size=a.shape), v, n


# A generator for stepping the agent with the environment
def play_games(a2c, envs, nsteps):
    s = envs.reset()
    for i in range(nsteps):
        a, _, _ = a2c.act(states)
        ns, r, d, _ = envs.step(a)
        yield i, s, a, r, ns, d
        s = ns
