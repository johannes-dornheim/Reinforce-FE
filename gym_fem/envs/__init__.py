from gym.envs.registration import register
from gym_fem.envs.deep_drawing import DeepDrawing
from gym_fem.envs.deep_drawing_13ts import DeepDrawingLong

register(
    id=DeepDrawing.ENV_ID,
    entry_point='gym_fem.envs:DeepDrawing',
    max_episode_steps=2500,
)

register(
    id=DeepDrawingLong.ENV_ID,
    entry_point='gym_fem.envs:DeepDrawingLong',
    max_episode_steps=2500,
)

