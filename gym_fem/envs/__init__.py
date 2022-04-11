from gym.envs.registration import register
from gym_fem.envs.deep_drawing import DeepDrawing, StressStateDeepDrawing, StressOffsetStateDeepDrawing
from gym_fem.envs.deep_drawing_13ts import DeepDrawingLong

register(
    id=DeepDrawing.ENV_ID,
    entry_point='gym_fem.envs:DeepDrawing',
    max_episode_steps=2500,
)

register(
    id=StressStateDeepDrawing.ENV_ID,
    entry_point='gym_fem.envs:StressStateDeepDrawing',
    max_episode_steps=2500,
)

register(
    id=StressOffsetStateDeepDrawing.ENV_ID,
    entry_point='gym_fem.envs:StressOffsetStateDeepDrawing',
    max_episode_steps=2500,
)