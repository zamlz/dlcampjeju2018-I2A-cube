
from gym.envs.registration import register

# 2x2x2 cube environment
register(
    id='cube-2x2x2-v0',
    entry_point='cube_gym.environment:CubeEnv',
    kwargs={'order': 2},
)

# 3x3x3 cube environment
register(
    id='cube-3x3x3-v0',
    entry_point='cube_gym.environment:CubeEnv',
    kwargs={'order': 3},
)
