import math

import numpy as np
from gym import spaces
from scipy import stats

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

    def __init__(self):

        self._simulation_id_templates = []
        for i in range(self.TIME_STEPS):
            bhf_string = '_'.join([f'{{BHF_{j}}}' for j in range(i + 1)])
            s = f'BHF-{bhf_string}__FRIC-{{FRIC}}'
            self._simulation_id_templates.append(s)

        self._current_conditions = None

        super().__init__()

    def step(self, action):
        action = self.action_values[action]
        self.info[f'bhf{self.time_step}'] = action
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
            friction = self._np_random_state.beta(1.75, 5)
            # scale
            friction = friction * 0.14
            # bin
            friction = math.ceil(friction / 0.014) * 0.014
            self._current_conditions = {'FRIC': friction}
        return self._current_conditions

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

    def _apply_reward_function(self, fem_results):
        """
        Args:
            fem_results (tuple): tuple of pandas dataframes for element-wise- and node-wise results
        Returns:
            observation (object): reward for given simulation results
        """
        element_data, node_data = fem_results

        v_mises_reward = 1.0 - self._calc_normalized_mises_l2(element_data)
        thickness_reward = self._calc_normalized_min_thickness(node_data)
        feeding_reward = 1.0 - self._calc_normalized_mean_feeding(node_data)

        self.info.update({'rt_v_mises': v_mises_reward,
                          'rt_thickness': thickness_reward,
                          'rt_feeding': feeding_reward})

        if any([v_mises_reward < 0.0, thickness_reward < 0.0, feeding_reward < 0.0]):
            return 0
        return (stats.hmean([v_mises_reward, thickness_reward, feeding_reward]) * 10) ** 2

    def _apply_observation_function(self, fem_results):
        """ to be implemented by special FEMEnv instance
        Args:
            fem_results (tuple): tuple of pandas dataframes for element-wise- and node-wise results
        Returns:
            observation (np.array): observation vector for given simulation results
        """
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

        self.info.update({f'ao_stamp_force{self.time_step}': stamp_force,
                          f'ao_blank_offset{self.time_step}': blank_offset,
                          f'ao_bh_offset{self.time_step}': bh_offset,
                          f'o_stamp_force{self.time_step}': o_stamp_force,
                          f'o_blank_offset{self.time_step}': o_blank_offset,
                          f'o_bh_offset{self.time_step}': o_bh_offset})

        return np.array([o_stamp_force, o_blank_offset, o_bh_offset])

    def _is_done(self):
        return self.time_step == self.TIME_STEPS - 1
