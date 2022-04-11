import time
from datetime import datetime
import shutil
import warnings
import os
import subprocess
from subprocess import check_output, TimeoutExpired, CalledProcessError
from abc import ABC
from pathlib import Path
import logging
import pandas as pd


class FEMWrapper(ABC):
    """ superclass for FEM-software specific wrapper-classes
    """

    def __init__(self, simulation_store_path):
        self.simulation_store_path = simulation_store_path

    def run_simulation(self, simulation_id, simulation_parameters, time_step, base_simulation_id=None):
        """ executes the simulation with given parameters and stores results under the given simulation-id
        Args:
            simulation_id (str): identifier of the requested simulation
            simulation_parameters (dict): named parameter-values for the parametric fem-simulation
            base_simulation_id (str): identifier of the basis simulation for the requested simulation
            time_step (int): current time-step
        """
        raise NotImplementedError

    def simulation_results_available(self, simulation_id):
        """ Returns true if given simulation has been calculated and results are available
        Args:
            simulation_id (str): identifier of the requested simulation
        Returns:
            results_available (bool): true if results are available, else false
        """
        raise NotImplementedError

    def read_simulation_results(self, simulation_id, root_simulation_id=None):
        """ Reads out results from the given simulation
        Args:
            simulation_id (str): identifier of the requested simulation
            root_simulation_id (str): identifier of the root simulation (first time-step)
        Returns:
            simulation_results (tuple): Tuple of two Pandas Dictionaries: (element-wise results, node-wise results)
        """
        raise NotImplementedError
    def is_terminal_state(self, simulation_id):
        """ returns True, if no former actions are possible based on the simulation-state
        Args:
            simulation_id (str): identifier of the requested simulation
        Returns:
            is_terminal (bool): True, if no former actions are possible based on the simulation-state
        """
        return False

    def request_lock(self, simulation_id, timeout_seconds=600):
        """ locks the given simulation (to enable parallel environments based on the same simulation-storage)
        Args:
            simulation_id (str): identifier of the simulation
            timeout_seconds (int): seconds from now until lock times out
        Returns:
            locked (bool): True if the lock has been successfully established, false if the lock was already set
        """
        job_directory = self.simulation_store_path.joinpath(f'{simulation_id}')
        job_directory.mkdir(parents=True, exist_ok=True)

        lock = job_directory.joinpath('.fem_wrapper_lock')
        try:
            lock.touch(exist_ok=False)
        except FileExistsError:
            lock_timeout = lock.read_text()
            if lock_timeout == '':
                logging.warning(f'empty lock: {lock}, setting lock-timeout t+600 and waiting')
                timeout = str(datetime.now().timestamp() + timeout_seconds)
                lock.write_text(timeout)
                return False
            elif datetime.now().timestamp() > float(lock_timeout):
                logging.warning(
                    f'overwriting timed out lock: {lock}, timeout: {datetime.fromtimestamp(float(lock_timeout))}')
            else:
                return False

        timeout = str(datetime.now().timestamp() + timeout_seconds)
        lock.write_text(timeout)
        return True

    def release_lock(self, simulation_id):
        """ Releases the simulation-lock
        Args:
            simulation_id (str): identifier of the requested simulation
        """
        job_directory = self.simulation_store_path.joinpath(f'{simulation_id}')
        lock = job_directory.joinpath('.fem_wrapper_lock')
        if lock.exists():
            lock.unlink()


class SimulationError(Exception):
    """ Thrown if simulation is running into numeric issues for the given parameters
    """

    def __init__(self):
        self.message = "Simulation-Model can not be solved for given parameters"


class AbaqusWrapper(FEMWrapper):
    def __init__(self, simulation_store_path, template_path, abaq_command, abaq_version_string,
                 cpu_count, abaq_timeout, reader_version, abq_forcekill=False, check_version=False):
        super().__init__(simulation_store_path)
        self.template_folder = template_path

        if abaq_command is not None:
            self.abaq_command = abaq_command
        else:
            if os.name == 'posix':
                self.abaq_command = 'abaqus'
            elif os.name == 'nt':
                self.abaq_command = r'C:\SIMULIA\Abaqus\Commands\abaqus.bat'
            else:
                raise OSError(f'No default abaqus command defined for OS: {os.name}')
        self.abaq_version_string = abaq_version_string
        self.cpu_count = cpu_count
        self.timeout = abaq_timeout
        self.abaq_reader_version = reader_version
        self.abaq_forcekill = abq_forcekill
        self.check_abq_version = check_version
        self.result_dict = {}
        self.NON_SIMABLE_ERR_MESSAGES = ["Abaqus/Standard Analysis exited with an error",
                                         "aborted with system error code 1073741819",
                                         "Analysis Input File Processor exited with an error"]
        self.CONNECTION_ERR_MESSAGES = ["IP address cannot be determined",
                                        "Failed to startup licensing"]

    def request_lock(self, simulation_id, timeout_seconds=600):
        return super().request_lock(simulation_id, timeout_seconds)

    def release_lock(self, simulation_id):
        super().release_lock(simulation_id)

    def run_simulation(self, simulation_id, simfolder_prepared=False,
                       simulation_parameters=None, time_step=None, base_simulation_id=None, visualize=True):
        assert simfolder_prepared or (simulation_parameters is not None and time_step is not None)
        time_step = time_step
        is_restart_simulation = base_simulation_id is not None

        job_directory = self.simulation_store_path.joinpath(f'{simulation_id}')

        if not simfolder_prepared:
            """ create simulation folder """
            if job_directory.exists():
                # unlink all files except lck and timeout file
                for file in job_directory.iterdir():
                    if file.stem not in ['.fem_wrapper_lock', '.timeout_log']:
                        if file.is_file():
                            file.unlink()
                        if file.is_dir():
                            shutil.rmtree(file)
            else:
                job_directory.mkdir(exist_ok=True)

            """ copy files from base-simulation folder """
            if is_restart_simulation:
                base_job_file_names = self._copy_restart_files(base_simulation_id, simulation_id)
            """ prepare fem-template """
            template_path = self.template_folder.joinpath(f'{time_step}.rlinp')
            input_file_path = job_directory.joinpath(f'{simulation_id}.inp')
            with open(template_path, "rt") as template:
                with open(input_file_path, "wt") as inp_file:
                    for line in template:
                        for param, value in simulation_parameters.items():
                            line = line.replace(f'<!{param}!>', str(value))
                        if '<!' in line:
                            raise ValueError(f"non defined parameter in job template: {line}")
                        inp_file.write(line)

        """ run simulation """
        run_cmd = [f'{self.abaq_command}', f'job={simulation_id}', 'interactive', f'cpus={self.cpu_count}']
        if is_restart_simulation:
            run_cmd.append(f'oldjob={base_simulation_id}')
        out = ""
        try_count = 0

        while f"Abaqus JOB {simulation_id} COMPLETED" not in out or \
                job_directory.joinpath(f'{simulation_id}.odb_f').exists():
            # delete all job-files except .inp
            for file in [f for f in job_directory.iterdir() if f.stem == simulation_id and f.suffix != '.inp']:
                try:
                    os.remove(file)
                except IOError as e:
                    logging.warning(f'simulation file {file} can not be deleted. {e}')

            # run Abaqus and wait for completion
            logging.info(f'RUN ABAQUS: {job_directory} || ' + ' '.join(run_cmd))
            # if n=10 timeouts reported: imeadiately raise simulation-error
            timeout_file = job_directory.joinpath('.timeout_log')
            if timeout_file.exists():
                n_timeouts = sum(1 for line in open(timeout_file))
                logging.info(f'simulation timed-out at the last {n_timeouts} attempts')
                if n_timeouts >= 5:
                    logging.warning(f'skipping simulation due to {n_timeouts} successive timeouts in previous attempts')
                    raise SimulationError

            try_count += 1
            try:
                start = time.time()
                if not self.abaq_forcekill:
                    out = check_output(run_cmd, cwd=job_directory, timeout=self.timeout, stderr=subprocess.STDOUT)
                    out = str(out)
                else:
                    # === dirty hack due to linux abq termination problem! ===
                    if 'interactive' in run_cmd:
                        run_cmd.remove('interactive')
                    out = check_output(run_cmd, cwd=job_directory, timeout=self.timeout, stderr=subprocess.STDOUT)
                    out = str(out)
                    analysis_finished = False
                    while not analysis_finished:
                        try:
                            with open(job_directory.joinpath(f'{simulation_id}.sta'), 'r') as f:
                                lines = f.read().splitlines()
                                analysis_finished = "COMPLETED SUCCESSFULLY" in lines[-2] + lines[-1]
                            time.sleep(1)
                        except IOError:
                            time.sleep(2)
                        if time.time() > start + self.timeout:
                            raise TimeoutError('abq timeout')

                    self._abaq_forcekill(job_directory, simulation_id)
                    out = f"Abaqus JOB {simulation_id} COMPLETED"
                    # === end of dirty hack ===
                    # remove timeout file (if timeout is followed by a solution in time)
                    if timeout_file.exists():
                        logging.info(f'successful simulation after {sum(1 for line in open(timeout_file))} '
                                     f'timed-out attemts')
                        timeout_file.unlink()
                logging.info(out)
                if 'Position in the queue' in out:
                    logging.warning("Abaqus licence-request was queued. " +
                                    f"Total time for solving: {time.time() - start} seconds.")
            except (TimeoutExpired, TimeoutError):
                if self.abaq_forcekill:
                    self._abaq_forcekill(job_directory, simulation_id)

                logging.warning(f'Abaqus Timed out after {self.timeout} seconds.')
                if try_count % 2 == 0:
                    # write timestamp to timeout file in experiment folder
                    open(timeout_file, 'a').write(f"{time.time()}\n")
                    raise SimulationError
            except CalledProcessError as e:
                if self.abaq_forcekill:
                    self._abaq_forcekill(job_directory, simulation_id)

                out = str(e.output).replace('\\r\\n', ' || ')
                logging.warning(f'Abaq. Error in try #{try_count}. output: {out}')

                # if wrong format of restart file: rerun base simulation and copy restart file
                if "The old-job restart file format is not recognized" in out:
                    logging.info(f'Trying to rerun base-simulation {base_simulation_id}')

                    # todo dirty base sim-id
                    if len(base_simulation_id) > 18:
                        i = re.search('_\d+.\d+__', base_simulation_id)
                        base_base_simulation_id = base_simulation_id[:i.start()] + base_simulation_id[(i.end()-2):]
                        self._copy_restart_files(base_base_simulation_id, base_simulation_id)

                        self.run_simulation(simulation_id=base_simulation_id, base_simulation_id=base_base_simulation_id,
                                            simfolder_prepared=True)
                    else:
                        self.run_simulation(simulation_id=base_simulation_id, simfolder_prepared=True)
                    self._copy_restart_files(base_simulation_id, simulation_id)

                # if Errors are simulation errors (not depending on Internet Connection / violated buffer etc.)
                if any([s in out for s in self.NON_SIMABLE_ERR_MESSAGES]) and \
                        not any([s in out for s in self.CONNECTION_ERR_MESSAGES]):
                    if try_count % 2 == 0:
                        raise SimulationError
                time.sleep(30)
            except Exception as e:
                if self.abaq_forcekill:
                    self._abaq_forcekill(job_directory, simulation_id)

                logging.warning(f'Unhandled Abaqus Error {e} abaqus output: {out}')
                time.sleep(10)
        """ clean base-job files """
        if is_restart_simulation and not simfolder_prepared:
            for file in [job_directory.joinpath(f) for f in base_job_file_names]:
                try:
                    os.remove(file)
                except OSError:
                    continue
    def _abaq_forcekill(self, job_directory, simulation_id):
        subprocess.call('pgrep standard | xargs kill', shell=True)
        abq_lock = job_directory.joinpath(f'{simulation_id}.lck')
        if abq_lock.exists():
            abq_lock.unlink()

    def _copy_restart_files(self, origin_simulation_id, target_simulation_id):
        i = 0
        while not self.request_lock(origin_simulation_id, 30):
            logging.warning(f'waiting for base-simulation lock release {origin_simulation_id}')
            time.sleep(2 ** i)
            i += 1
        job_directory = self.simulation_store_path.joinpath(target_simulation_id)
        base_job_directory = self.simulation_store_path.joinpath(origin_simulation_id)
        base_job_file_names = [f'{origin_simulation_id}.{ext}' for ext in ['res', 'prt', 'mdl', 'stt', 'odb']]
        base_job_file_paths = [base_job_directory.joinpath(f) for f in base_job_file_names]
        for file in base_job_file_paths:
            if file.exists():
                while True:
                    try:
                        shutil.copy(file,
                                    job_directory.joinpath(file.name))
                    except PermissionError:
                        logging.warning(f'Permission denied for file {file}!')
                        time.sleep(30)
                        continue
                    break
            else:
                raise IOError(f'file not existent: {file} (required for restart-simulation)')

        self.release_lock(origin_simulation_id)
        return base_job_file_names

    def simulation_results_available(self, simulation_id):
        if simulation_id in self.result_dict.keys():
            return True

        job_directory = self.simulation_store_path.joinpath(f'{simulation_id}')
        odb_file = job_directory.joinpath(f'{simulation_id}.odb')
        msg_file = job_directory.joinpath(f'{simulation_id}.msg')

        if odb_file.exists() and msg_file.exists():
            # check for lock-file or job ended with analysis-error, if True: simulation was aborted
            msg_contents = msg_file.read_text()
            if job_directory.joinpath(f'{simulation_id}.lck').exists() or \
                    "THE ANALYSIS HAS BEEN COMPLETED" not in msg_contents:
                logging.warning(f'incomplete simulation, no results available!')
                return False

            # check for abaqus version used at solve-time
            if self.check_abq_version:
                with open(msg_file, 'r') as msg:
                    msg_header = msg.read(500)
                    if self.abaq_version_string not in msg_header:
                        warnings.warn(f"different abaq version used for {simulation_id}")
                        return False
            return True
        return False

    def read_simulation_results(self, simulation_id, root_simulation_id=None):
        if simulation_id in self.result_dict.keys():
            return self.result_dict[simulation_id]

        job_directory = self.simulation_store_path.joinpath(f'{simulation_id}')

        node_csv = job_directory.joinpath(f'node_extract_{self.abaq_reader_version}.csv')
        element_csv = job_directory.joinpath(f'element_extract_{self.abaq_reader_version}.csv')

        # read out results from odb if not already done
        if not (node_csv.exists() and element_csv.exists()):
            # create reader-script command
            reader_script = 'AbaqReader.py'
            odb_path = job_directory.joinpath(f'{simulation_id}.odb')
            read_cmd = [f'{self.abaq_command}', 'python', reader_script, f'{odb_path}', f'{job_directory}']
            i = 1
            if root_simulation_id is not None:
                while not self.request_lock(root_simulation_id, timeout_seconds=120):
                    logging.warning(f'waiting for root-simulation lock release {root_simulation_id}')
                    time.sleep(2 ** i)
                    i += 1

                root_job_directory = self.simulation_store_path.joinpath(root_simulation_id)
                root_odb_path = root_job_directory.joinpath(f'{root_simulation_id}.odb')
                read_cmd += ['--first_odb_path', f'{root_odb_path}']

            # execute reader
            logging.info(' '.join(read_cmd))
            try:
                check_output(read_cmd, cwd=str(Path(__file__).parent.joinpath('simulation_scripts')),
                             stderr=subprocess.STDOUT)
            except BaseException as e:
                logging.warning(e)
                logging.warning(e.output)
                if "odb is from a more recent release of Abaqus" in e.output.decode("utf-8"):
                    logging.info(f"trying to rerun simulation {simulation_id} and root-simulation {root_simulation_id} !")

                    # todo dirty base sim-id
                    i = re.search('_\d+.\d+__', simulation_id)
                    base_simulation_id = simulation_id[:i.start()] + simulation_id[(i.end()-2):]
                    self._copy_restart_files(base_simulation_id, simulation_id)
                    self.run_simulation(simulation_id, base_simulation_id=base_simulation_id,
                                        simfolder_prepared=True, visualize=False)

                    self.request_lock(root_simulation_id)
                    self.run_simulation(root_simulation_id, simfolder_prepared=True, visualize=False)
                    self.release_lock(root_simulation_id)
                    return self.read_simulation_results(simulation_id, root_simulation_id=root_simulation_id)

            if root_simulation_id is not None:
                self.release_lock(root_simulation_id)

        # import and return results
        node_data = pd.read_csv(node_csv, sep='\s*,\s*', index_col=None, engine='python')
        element_data = pd.read_csv(element_csv, sep='\s*,\s*', index_col=None, engine='python')

        self.result_dict[simulation_id] = (element_data, node_data)
        return element_data, node_data

    def get_state_visualization(self, simulation_id, *args, **kwargs):
        job_directory = self.simulation_store_path.joinpath(f'{simulation_id}')
        img_path = job_directory.joinpath(f'{simulation_id}.png')
        if not img_path.exists():
            try:
                self._plot(simulation_id)
            except SystemError:
                img_path = None
        return img_path

    def _plot(self, simulation_id):
        if simulation_id is None:
            raise SystemError

        job_directory = self.simulation_store_path.joinpath(f'{simulation_id}')
        odb_path = job_directory.joinpath(f'{simulation_id}.odb')
        img_path = job_directory.joinpath(f'{simulation_id}.png')

        try:
            out = check_output([self.abaq_command, 'cae', 'noGUI=PlotLastFrame.py', '--', odb_path, img_path],
                               cwd=str(Path(__file__).parent.joinpath('simulation_scripts')))
            out = str(out)
            logging.info(out)
        except CalledProcessError as e:
            logging.warning(f"State-Visualization not possible, CalledProcessError in visualization-script: {e}")
            raise SystemError
