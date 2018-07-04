
from gym.envs.registration import register

# 2x2x2 cube environment
register(
    id='cube-x2-v0',
    entry_point='cube_gym.environment:CubeEnv',
    kwargs={'order': 2},
)

# 3x3x3 cube environment
register(
    id='cube-x3-v0',
    entry_point='cube_gym.environment:CubeEnv',
    kwargs={'order': 3},
)

# 4x4x4 cube environment
register(
    id='cube-x4-v0',
    entry_point='cube_gym.environment:CubeEnv',
    kwargs={'order': 4},
)
