from gym_fem.fem_wrapper import AbaqusWrapper, SimulationError
from gym import Wrapper

import math
from pathlib import Path
import numpy as np
from gym import spaces
from scipy import stats
from scipy.signal import convolve2d
from gym_fem.fem_env import FEMEnv


class DeepDrawing(FEMEnv):
    ENV_ID = '2d-deepdrawing-5ts-v2'

    action_names = ['BHF']
    action_space = spaces.Discrete(7)

    observation_space = spaces.Box(-np.inf, np.inf, shape=[3], dtype='float32')
    fem_engine = "Abaq"

    TIME_STEPS = 5
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

    _standard_img_size = (731,914,3)
    _standard_img = np.zeros(_standard_img_size, dtype=np.uint8)

    _SIM_ERROR_REWARD = 0.0

    def __init__(self):

        self._simulation_id_templates = []
        for i in range(self.TIME_STEPS):
            bhf_string = '_'.join([f'{{BHF_{j}}}' for j in range(i + 1)])
            s = f'BHF-{bhf_string}__FRIC-{{FRIC}}'
            self._simulation_id_templates.append(s)

        self._current_conditions = None

        super().__init__()

        abaq_params = self.config['abaqus parameters']
        if self.SIM_BASIS is not None:
            template_folder = Path(__file__).parent.parent.joinpath(f'assets/abaqus_models/{self.SIM_BASIS}')
        else:
            template_folder = Path(__file__).parent.parent.joinpath(f'assets/abaqus_models/{self.ENV_ID}')

        solver_path = abaq_params.get('solver_path')
        if solver_path in ['', 'None']:
            solver_path = None

        self._fem_wrapper = AbaqusWrapper(self.sim_storage,
                                          template_folder,
                                          solver_path,
                                          abaq_params.get('abaq_version'),
                                          abaq_params.getint('cpu_kernels', fallback=4),
                                          abaq_params.getint('timeout', fallback=300),
                                          abaq_params.getint('reader_version', fallback=0),
                                          abq_forcekill=abaq_params.getboolean('forcekill', fallback=False),
                                          check_version=abaq_params.getboolean('check_sim_result_abq_version', fallback=False))
        self._bypassed_fric = None

    def step(self, action):
        action = self.action_values[action]
        self.info[f'bhf'] = action
        return super().step(action)

    def reset(self):
        super().reset()
        return np.zeros(3)

    def _sample_process_conditions(self):
        """ to be implemented by special FEMEnv instance
        Returns:
            process-conditions (dict):
                dictionary with process-conditions (Keys used have to be identical with abaq-template keys)
        """
        if self.time_step == 0:
            if self._bypassed_fric is not None:
                friction = self._bypassed_fric
            else:
                friction = self._np_random_state.beta(1.75, 5)
                # scale
                friction = friction * 0.14
                # bin
                friction = math.ceil(friction / 0.014) * 0.014
            self._current_conditions = {'FRIC': friction}
        return self._current_conditions

    def bypass_fric(self, fric):
        # Enables bypassing friction-values for data-samping purposes
        self._bypassed_fric = fric

    def _calc_normalized_mises_l2(self, element_data):
        """
        @param element_data: EM element output format, required: MISES
        @type element_data: pandas.DataFrame
        @return: l2 norm of v. mises stresses for given data
        @rtype: float_
        """
        v_mises_stresses = list(element_data['MISES'])
        return (np.linalg.norm(v_mises_stresses, ord=2) - self._MIN_MISES_L2) / float(
            self._MAX_MISES_L2 - self._MIN_MISES_L2)

    def _calc_normalized_mean_feeding(self, node_data):
        """
        @param node_data: FEM node output format, required: INSTANCE, NODE_ID, X_OFFSET
        @type node_data: pandas.DataFrame
        @return: mean feeding-length for given data
        @rtype: float
        """
        feeding = list(node_data[(node_data['INSTANCE'] == 'BLECH') &
                                 (node_data['NODE_ID'].isin(self._BLANK_RIGHTMOST_NODES))]['X_OFFSET'])
        return (np.mean(feeding) - self._MIN_FEEDING) / (self._MAX_FEEDING - self._MIN_FEEDING)

    def _calc_normalized_min_thickness(self, node_data):
        """
        @param node_data: FEM node output format, required: INSTANCE, NODE_ID, X_COORD, Y_COORD
        @type node_data: pandas.DataFrame
        @return: blank thickness on thinnest spot
        @rtype: float
        """
        min_thickness = np.inf
        for a, b in self._BLANK_THICKNESS_NODE_PAIRS:
            node_a = node_data[(node_data['INSTANCE'] == 'BLECH') & (node_data['NODE_ID'] == a)]
            node_b = node_data[(node_data['INSTANCE'] == 'BLECH') & (node_data['NODE_ID'] == b)]
            diff_vec = (float(node_a['X_COORD']) - float(node_b['X_COORD']),
                        float(node_a['Y_COORD']) - float(node_b['Y_COORD']))
            min_thickness = min(np.linalg.norm(diff_vec), min_thickness)

        return (min_thickness - self._MIN_THICKNESS_L_NEGINF) / (
                self._MAX_THICKNESS_L_NEGINF - self._MIN_THICKNESS_L_NEGINF)

    def _apply_reward_function(self, fem_results, done):
        """
        Args:
            fem_results (tuple): tuple of pandas dataframes for element-wise- and node-wise results
        Returns:
            observation (object): reward for given simulation results
        """
        if not done:
            self.info.update({'rt_v_mises': 0.0,
                              'rt_thickness': 0.0,
                              'rt_feeding': 0.0})
            return 0.0

        element_data, node_data = fem_results

        v_mises_reward = 1.0 - self._calc_normalized_mises_l2(element_data)
        thickness_reward = self._calc_normalized_min_thickness(node_data)
        feeding_reward = 1.0 - self._calc_normalized_mean_feeding(node_data)

        self.info.update({'rt_v_mises': v_mises_reward,
                          'rt_thickness': thickness_reward,
                          'rt_feeding': feeding_reward})

        if any([v_mises_reward < 0.0, thickness_reward < 0.0, feeding_reward < 0.0]):
            return 0
        return stats.hmean([v_mises_reward, thickness_reward, feeding_reward]) * 10

    def _apply_observation_function(self, fem_results):
        """ to be implemented by special FEMEnv instance
        Args:
            fem_results (tuple): tuple of pandas dataframes for element-wise- and node-wise results
        Returns:
            observation (np.array): observation vector for given simulation results
        """
        if fem_results is None:
            return np.zeros(3)
        element_data, node_data = fem_results
        """ stamp force """
        stamp_force = node_data[node_data['INSTANCE'] == 'STEMPEL']['TOTAL_FORCE_2'].values[0]
        o_stamp_force = (self._np_random_state.normal(stamp_force, self._STAMP_FORCE_STDV, 1)[0])

        """ blank offset """
        # mean offset in x direction for rightmost nodes
        blank_offset = node_data[node_data['INSTANCE'] == 'BLECH']['X_OFFSET'].mean()
        o_blank_offset = (self._np_random_state.normal(blank_offset, self._BLANK_OFFSET_STDV, 1)[0])

        """ blank holder offset """
        bh_offset = node_data[node_data['INSTANCE'] == 'NIEDERHALTER']['Y_OFFSET'].values[0]
        clipped_bh_offset = max(bh_offset, -0.25)
        o_bh_offset = (self._np_random_state.normal(clipped_bh_offset, self._BH_OFFSET_STDV, 1)[0])

        self.info.update({f'ao_stamp_force': stamp_force,
                          f'ao_blank_offset': blank_offset,
                          f'ao_bh_offset': bh_offset,
                          f'o_stamp_force': o_stamp_force,
                          f'o_blank_offset': o_blank_offset,
                          f'o_bh_offset': o_bh_offset})

        return np.array([o_stamp_force, o_blank_offset, o_bh_offset])

    def _is_done(self,  state=None):
        return self.time_step == self.TIME_STEPS - 1


class StressStateDeepDrawing(DeepDrawing):
        # The state is defined by the v. Mises Stress matrix and the time-step instead of abstract observable values
        ENV_ID = '2d-deepdrawing-5ts-stressstate-v0'
        SIM_BASIS = '2d-deepdrawing-5ts-v2'
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5, 80, 1), dtype=np.float32)

        # calculated based on 56501 episodes
        MIN_MISES = 0
        MAX_MISES = 890.072937012

        def _apply_observation_function(self, fem_results):
            if fem_results == None:
                return np.zeros((5, 80, 1))
            element_data, node_data = fem_results
            # read out blank stresses
            mises = element_data['MISES']
            # Bring into form of a 2d-field
            assert len(mises) == 400
            mises = np.array(mises).reshape((5, 80))
            mises = (mises - self.MIN_MISES) / (self.MAX_MISES - self.MIN_MISES) * 256
            return mises

        def reset(self):
            super().reset()
            return np.zeros(shape=(5, 80, 1))


class StressOffsetStateDeepDrawing(DeepDrawing):
        # The state is defined by the v. Mises Stress matrix, the x- and the y- offsets
        ENV_ID = '2d-deepdrawing-5ts-stressoffsetstate-v0'
        SIM_BASIS = '2d-deepdrawing-5ts-v2'
        # calculated based on 56501 episodes
        MIN_MISES = 0
        MAX_MISES = 890.072937012
        MIN_X_OFFSET = -2.041515
        MAX_X_OFFSET = 12.419274999999999
        MIN_Y_OFFSET = -1.005121
        MAX_Y_OFFSET = 25.002325000000003

        observation_space = spaces.Box(low=0, high=255, shape=(5, 80, 3), dtype=np.float32)

        def _apply_observation_function(self, fem_results):
            if fem_results == None:
                return np.zeros((5, 80, 3))
            element_data, node_data = fem_results
            # read out blank stresses
            mises = element_data['MISES']
            # Bring into form of a 2d-field
            assert len(mises) == 400
            mises = np.array(mises).reshape((5, 80))[:, ::-1]

            # x offset
            blank_nodes = node_data[node_data['INSTANCE'] == 'BLECH']
            x_offset = np.array(blank_nodes['X_OFFSET']).reshape((6, 81))[:, ::-1]
            x_offset = convolve2d(x_offset, np.ones((2, 2)) * 0.25, mode='valid')
            # y offset
            y_offset = np.array(blank_nodes['Y_OFFSET']).reshape((6, 81))[:, ::-1]
            y_offset = convolve2d(y_offset, np.ones((2, 2)) * 0.25, mode='valid')

            # scale and clip to 0,255 to be used with standard drl frameworks
            mises = (mises - self.MIN_MISES) / (self.MAX_MISES - self.MIN_MISES) * 256
            x_offset = (x_offset - self.MIN_X_OFFSET) / (self.MAX_X_OFFSET - self.MIN_X_OFFSET) * 256
            y_offset = (y_offset - self.MIN_Y_OFFSET) / (self.MAX_Y_OFFSET - self.MIN_Y_OFFSET) * 256
            o = np.moveaxis(np.stack([mises, x_offset, y_offset]), 0, -1)
            return o

        def reset(self):
            super().reset()
            return np.zeros(shape=(5, 80, 3))

class DeepDrawingMORLWrapper(Wrapper):
    # MORL environment
    def __init__(self, env, r_terms, prohibit_negative_rewards=True):
        super().__init__(env)
        for r_term in r_terms:
            assert r_term in ['rt_feeding', 'rt_thickness', 'rt_v_mises']
        self.r_terms = r_terms
        self.prohibit_negatives = prohibit_negative_rewards

    def step(self, action):
        o, r, done, info = self.env.step(action)
        r = np.array([info[r_term] for r_term in self.r_terms])
        if self.prohibit_negatives and np.any(r < 0):
            r = np.zeros(len(self.r_terms))
        return o, r, done, info

