
from actor_critic.trainer import train, ActorCritic
from actor_critic.pd_test import pd_test
from actor_critic.policy  import A2CBuilder 

import numpy as np

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
        # returns a, v, n
        if np.random.sample() > self.epsilon:
            return self.step_model.step(obs, stochastic=stochastic)
        else:
            return np.random.randint(self.nact, size=obs.shape[0]), v, n

