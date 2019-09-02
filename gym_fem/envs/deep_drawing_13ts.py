import numpy as np

from gym_fem.envs.deep_drawing import DeepDrawing


class DeepDrawingLong(DeepDrawing):
    ENV_ID = '2d-deepdrawing-13ts-v2'

    fem_engine = "Abaq"

    TIME_STEPS = 13
    action_values = np.linspace(2, 14, 7)
    # limits, used for reward-term normalization
    # empiric values based on 100 randomly parametrized simulations
    _MIN_THICKNESS_L_NEGINF = 1.85
    _MAX_THICKNESS_L_NEGINF = 2.13
    _MIN_MISES_L2 = 3950.41
    _MAX_MISES_L2 = 4331.56
    _FRACTURE_DIST = 1.0
    _MIN_FEEDING = 9.09
    _MAX_FEEDING = 10.74

    # observation noise stdv (empirical range * assumed measurement accuracy)
    _STAMP_FORCE_STDV = 141450 * 0.01
    _BLANK_OFFSET_STDV = 3.7 * 0.005
    _BH_OFFSET_STDV = 0.25 * 0.01

    # fem-model specific node-ids, used for reward-calculation
    _BLANK_THICKNESS_NODE_PAIRS = [(a, b) for a, b in zip(range(1, 82), range(406, 487))]
    _BLANK_RIGHTMOST_NODES = [1, 82, 163, 244, 325, 406]
