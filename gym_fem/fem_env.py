from gym_fem.fem_wrapper import AbaqusWrapper, SimulationError
from gym_fem.helpers import CSVLogger

import gym
from gym.utils import seeding
from gym.core import Wrapper

import numpy as np
from collections.abc import Iterable
from abc import ABC
from pathlib import Path
import configparser
import inspect
import imageio
import time
import logging
import shutil

class FEMEnv(gym.Env, ABC):
    """ abstract class for fem-based environments
    unset class variables have to be set by specific environment
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    # environment-id: used as GYM-environment name and for template-paths and simulation-storage-paths
    ENV_ID = None

    # readable names for individual actions, used for solver-templates
    action_names = []

    # Gym Spaces
    action_space = None
    observation_space = None

    # per time-step string templates for the simulation-id (uses self.simulation_parameters)
    _simulation_id_templates = None

    # FEM-engine used (e.g. "Abaq")
    fem_engine = None

    # reward given if the simulation is not solvable
    _not_solvable_reward = 0

    # img used for rgb-, and human rendering when no image is available
    _standard_img = np.zeros((731, 914, 3), dtype=np.uint8)

    def __init__(self):
        # current episode
        self.episode = 0
        # current time-step
        self.time_step = 0

        # data returned by step(self, action) for 'FEMLogger' or for special purposes like multiobjective-Learning
        # label prefix conventions to enable generic visualization / agents etc.:
        # rt: reward-term
        # ao: actual observation (without artificial additive noise)
        self.info = {}

        # parameters filled into the solver-template before simulation and used to set the simulation-id
        self.simulation_parameters = {}

        # used for restart-simulations
        self._root_simulation_id = None
        self._base_simulation_id = None

        # stochastic environment dynamic has to be derived from this
        self._np_random_state = None
        self.seed()

        self.viewer = None
        self.state_img_path = None

        # read config
        config = configparser.ConfigParser()
        config.read(Path(__file__).parent.joinpath('config.ini'))
        general_config = config['general parameters']

        self.persistent_simulation = general_config.getboolean('persistent_simulation')
        self.visualize = general_config.getboolean('visualize')

        storage = general_config.get('simulation_storage', fallback=None)

        # determine paths for current environment
        if self.persistent_simulation:
            if storage in [None, '', 'None']:
                self.sim_storage = Path(__file__).parent.joinpath(f'sim_storage/{self.ENV_ID}')
            else:
                self.sim_storage = Path(f'{storage}/{self.ENV_ID}')
        else:
            i = 0
            self.sim_storage = Path(f'{storage}/tmp/{self.ENV_ID}_{i}')
            while self.sim_storage.exists():
                i += 1
                self.sim_storage = Path(f'{storage}/tmp/{self.ENV_ID}_{i}')
            logging.info(f'persistence off, creating temporary simulation-storage: {self.sim_storage}')


        self.sim_storage.mkdir(exist_ok=True, parents=True)

        # read abaqus specific config
        if self.fem_engine == 'Abaq':
            abaq_params = config['abaqus parameters']

            template_folder = Path(__file__).parent.joinpath(f'assets/abaqus_models/{self.ENV_ID}')

            solver_path = abaq_params.get('solver_path')
            if solver_path in ['', 'None']:
                solver_path = None

            self.fem_wrapper = AbaqusWrapper(self.sim_storage,
                                             template_folder,
                                             solver_path,
                                             abaq_params.get('abaq_version'),
                                             abaq_params.getint('cpu_kernels', fallback=4),
                                             abaq_params.getint('timeout', fallback=300),
                                             abaq_params.getint('reader_version', fallback=0))
        else:
            raise NotImplementedError

    def step(self, action):
        """
        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) :
                amount of reward returned after previous action
            done (boolean):
                whether the episode has ended, in which case further step() calls will return undefined results
            info (dict):
                contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # update simulation-parameters by the current action (supports multiple-input control cases)
        if isinstance(action, Iterable):
            for i, a in enumerate(action):
                self.simulation_parameters[f'{self.action_names[i]}_{self.time_step}'] = a
        else:
            self.simulation_parameters[f'{self.action_names[0]}_{self.time_step}'] = action

        # read out current process conditions
        process_conditions = self._sample_process_conditions()
        self.simulation_parameters.update(process_conditions)
        self.info.update(process_conditions)

        # create simulation-id (for file- and folder-names) from simulation-parameters
        simulation_id = self._simulation_id_templates[self.time_step].format(**self.simulation_parameters)
        logging.debug(simulation_id)
        i = 1

        # wait while simulation is locked
        while not self.fem_wrapper.request_lock(simulation_id):
            logging.warning(f'waiting for lock release {simulation_id}')
            time.sleep(2 ** i)
            i += 1

        try:
            # check simulation store for results, if none available: simulate
            if self.fem_wrapper.simulation_results_available(simulation_id):
                pass
            else:
                # run simulation
                self.fem_wrapper.run_simulation(simulation_id,
                                                self.simulation_parameters,
                                                self.time_step,
                                                self._base_simulation_id)
        except SimulationError:
            logging.warning(f'{simulation_id} not solvable!')
            o = np.zeros(3)
            r = 0
            done = True
        else:
            # read FEM-results
            fem_results = self.fem_wrapper.read_simulation_results(simulation_id,
                                                                   root_simulation_id=self._root_simulation_id)

            # apply reward function
            r = self._apply_reward_function(fem_results)
            # apply observation function
            o = self._apply_observation_function(fem_results)
            # visualize
            if self.visualize:
                self.state_img_path = self.fem_wrapper.get_state_visualization(simulation_id)
                self.render()
            done = self._is_done()

        if self.time_step == 0:
            self._root_simulation_id = simulation_id
        self._base_simulation_id = simulation_id

        self.time_step += 1
        if done:
            episode_string = f'{self.episode}: Reward {r}, Trajectory {simulation_id}'
            # print(colorize(episode_string, 'green', bold=True))
            logging.info(episode_string)

        self.fem_wrapper.release_lock(simulation_id)

        return o, r, done, self.info

    def _apply_reward_function(self, fem_results):
        """ to be implemented by special FEMEnv instance (use random numbers seeded in seed())
        Args:
            fem_results (tuple): tuple of pandas dataframes for element-wise- and node-wise results
        Returns:
            observation (object): reward for given simulation results
        """
        raise NotImplementedError

    def _apply_observation_function(self, fem_results):
        """ to be implemented by special FEMEnv instance (use random numbers seeded in seed())
        Args:
            fem_results (tuple): tuple of pandas dataframes for element-wise- and node-wise results
        Returns:
            observation (object): observation vector for given simulation results
        """
        raise NotImplementedError

    def _sample_process_conditions(self):
        """ to be implemented by special FEMEnv instance (uses random numbers seeded in seed())
        Returns:
            process-conditions (dict):
                dictionary with process-conditions (Keys used have to be identical with abaq-template keys)
        """
        raise NotImplementedError

    def _is_done(self):
        """ to be implemented by special FEMEnv instance, returns True if the current State is a terminal-state
        Returns:
            done (bool):
                dictionary with process-conditions (Keys used have to be identical with abaq-template keys)
        """
        raise NotImplementedError

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        self.simulation_parameters = {}
        self.time_step = 0
        self.info = {}
        self._base_simulation_id = None
        self._root_simulation_id = None
        self.episode += 1

    def render(self, mode='human'):
        """Renders the environment.

        - human: render to the current display or terminal seed and
          return nothing. Usually for human consumption.

        Args:
            mode (str): the mode to render with
        """
        if self.visualize:
            if self.state_img_path is None:
                return self._standard_img

            img = imageio.imread(self.state_img_path, pilmode='RGB')
            if mode == 'rgb_array':
                return img
            if mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer(maxwidth=1000)
                self.viewer.imshow(img)
                return self.viewer.isopen

        super(FEMEnv, self).render(mode=mode)

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:Solver
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self._np_random_state, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if not self.persistent_simulation:
            shutil.rmtree(self.sim_storage)

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

"""
class PseudoContinuousActions(ActionWrapper):
    '''
    FEM simulations usually are computationally expensive. For the reuse of previous simulation-results,
    the continuous actions are discretized artificially. To enable the application of continuous algorithms,
    by using this wrapper the discretization-step is intransparent for the agent.
    '''

    def __init__(self, env):
        assert (type(env) in inspect.getmro(type(env))), \
            f"PseudoContinuousActions Wrapper is defined for Environments of type FEMEnv, " \
                "given: {inspect.getmro(type(env))}"

        logging.warning("PseudoContinuousActions Wrapper overwrites the environments action-space")
        env.action_space = spaces.Box(min(env.action_values), max(env.action_values))

        super().__init__(env)

    def action(self, action):
        return (np.abs(self.action_values - action)).argmin()
"""

class FEMCSVLogger(Wrapper):
    """
    Specific csv-file logger for fem-environments. Complements the default gym monitor / stats_recorder.
    """

    def __init__(self, env, outdir):
        assert (type(env) in inspect.getmro(type(env))), \
            f"FEMLogger Wrapper is defined for Environments of type FEMEnv, " \
            f"given: {inspect.getmro(type(env))}"
        self._iteration_start = time.time()
        self._accumulated_reward = 0

        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        log_file = outdir.joinpath('env_log.csv')
        if log_file.exists():
            logging.warning(f'{log_file} already existent!')
        self._logger = CSVLogger(log_file)

        super().__init__(env)

    def step(self, action):
        o, r, done, info = super().step(action)
        self._accumulated_reward += r

        try:
            for i, a in enumerate(action):
                self._logger.set_value(f'action{i}_{self.time_step}', a)
        except TypeError:
            self._logger.set_value(f'action{self.time_step}', action)
        self._logger.set_value(f'reward{self.time_step}', r)
        self._logger.set_values(info)

        if done:
            self._set_episode_log_vals()
            self._logger.write_log()

        return o, r, done, info

    def reset(self, **kwargs):
        if self.episode > 0:
            self._accumulated_reward = 0
            self._iteration_start = time.time()
        return self.env.reset(**kwargs)

    def close(self):
        self._logger.write_log()
        return super().close()

    def _set_episode_log_vals(self):
        runtime = time.time() - self._iteration_start
        self._logger.set_value('iteration', int(self.episode))
        self._logger.set_value('runtime', runtime)
        self._logger.set_value('reward', self._accumulated_reward)
