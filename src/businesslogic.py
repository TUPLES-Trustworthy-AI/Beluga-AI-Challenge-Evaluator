import json
import os
import time
import zipfile
from io import BytesIO
from http.client import RemoteDisconnected
from web_service.competitor_api_utils import call_setup_api, call_plan_api, call_setup_problem_api, \
    call_start_simulation_api, call_next_action_api

import glob
import shutil
import numpy as np

# Add the tools directory to the path, to ensure consistent import names
import sys
sys.path.insert(0, os.path.join('/app', 'src', 'tools'))

from skd_domains.skd_pddl_domain import SkdPDDLDomain
from skd_domains.skd_spddl_domain import SkdSPDDLDomain
from beluga_lib.beluga_problem import BelugaProblemDecoder, BelugaProblemEncoder
from evaluation.evaluators import EvaluationSupport
from evaluation.evaluators import SingleSimulationOutcome, MultipleSimulationOutcome
from evaluation.planner_api import BelugaAction, BelugaPlan
from evaluation.evaluators import EvaluationException, InvalidActionException
from evaluation.planner_api import BelugaPlan, ProbabilisticPlanningMetatada
from evaluation.planner_api import action_from_json_obj

import tomllib
import logging

# Setup logging system
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG)

# Default parameters
class Configuration:
    def __init__(self):
        # Fixed options (these cannot be configured and should not be changed)
        self.problem_dir = os.path.join('..', 'benchmarks', 'scalability_challenge', 'deterministic', 'training')
        self.problem_dir_prob = os.path.join('..', 'benchmarks', 'scalability_challenge', 'probabilistic', 'arrivals', 'training')
        self.input_dir = os.path.join('..', 'res', 'input')
        self.output_dir = os.path.join('..', 'res', 'output')
        self.problem_file_name = 'problem.json'
        self.plan_file_name = 'plan.json'
        self.state_and_metadata_name = 'state_and_metadata.json'
        self.action_file_name = 'action.json'
        self.msg_no_plan = 'No plan was produced'
        self.msg_timeout = 'Plan not accepted (time limit reached)'

        # Configurable options (edit the "conf.toml" file to change these)
        self.seed = None
        self.time_limit = 15 * (60+10) # 15 minutes, plus 10 sec of overhead buffer, by default
        self.max_steps = 20
        self.nsamples = 30
        self.alpha = 0.7
        self.beta = 0.0004


# Function to read configuration options
def conf_from_file():
    # Define the configuration file path
    conf_file_path = os.path.join('..', 'conf.toml')
    # Build a configuration object
    res = Configuration()
    # Attempt to read configuration options from a file
    try:
        with open(conf_file_path, 'rb') as fp:
            conf_data = tomllib.load(fp)
        if 'problem_dir' in conf_data['evaluation']:
            res.problem_dir = conf_data['evaluation']['problem_dir']
        if 'problem_dir_prob' in conf_data['evaluation']:
            res.problem_dir_prob = conf_data['evaluation']['problem_dir_prob']
        if 'seed' in conf_data['evaluation']:
            res.seed = conf_data['evaluation']['seed']
        if 'time_limit' in conf_data['evaluation']:
            res.time_limit = conf_data['evaluation']['time_limit']
        if 'max_steps' in conf_data['evaluation']:
            res.max_steps = conf_data['evaluation']['max_steps']
        if 'nsamples' in conf_data['evaluation']:
            res.nsamples = conf_data['evaluation']['nsamples']
        if 'alpha' in conf_data['evaluation']:
            res.alpha = conf_data['evaluation']['alpha']
        if 'beta' in conf_data['evaluation']:
            res.beta = conf_data['evaluation']['beta']

    except Exception:
        logging.error('Cannot read configuration file, using default configuration options')

    return res


# Read configuration options
configuration = conf_from_file()


def get_output_dir(submission_id=None, problem_id=None, simulation_id=None, step_id=None):
    output_dir = configuration.output_dir
    if submission_id is not None:
        output_dir = os.path.join(output_dir, f'submission_id={submission_id}')
    if problem_id is not None:
        output_dir = os.path.join(output_dir, f'problem_id={problem_id}')
    if simulation_id is not None:
        output_dir = os.path.join(output_dir, f'simulation_id={simulation_id}')
    if step_id is not None:
        output_dir = os.path.join(output_dir, f'step_id={step_id}')
    return output_dir


def get_input_dir(submission_id=None, problem_id=None, simulation_id=None, step_id=None):
    input_dir = configuration.input_dir
    if submission_id is not None:
        input_dir = os.path.join(input_dir, f'submission_id={submission_id}')
    if problem_id is not None:
        input_dir = os.path.join(input_dir, f'problem_id={problem_id}')
    if simulation_id is not None:
        input_dir = os.path.join(input_dir, f'simulation_id={simulation_id}')
    if step_id is not None:
        input_dir = os.path.join(input_dir, f'step_id={step_id}')
    return input_dir


def clear_tmp_dirs(submission_id=None, problem_id=None, simulation_id=None, step_id=None):
    # Clear the output directory
    output_dir = get_output_dir(submission_id, problem_id, simulation_id, step_id)
    shutil.rmtree(output_dir, ignore_errors=True)
    # Clear the input directory
    input_dir = get_input_dir(submission_id, problem_id, simulation_id, step_id)
    shutil.rmtree(input_dir, ignore_errors=True)


def write_json_message(name, content, submission_id=None, problem_id=None, simulation_id=None, step_id=None):
    # Determine the output directory
    output_dir = get_output_dir(submission_id, problem_id, simulation_id, step_id=step_id)
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    # Write the message
    with open(os.path.join(output_dir, f'{name}.json'), 'w') as fp:
        json.dump(content, fp)


def zip_json_file(json_file_path, in_archive_name):
    # Create an in-memory ZIP file containing the JSON
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(json_file_path, in_archive_name)  # Add JSON file to ZIP
    zip_buffer.seek(0)
    # Return the zip buffer
    return zip_buffer


def zip_json_object(json_obj, in_archive_name):
    # Create an in-memory ZIP file containing the JSON
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(in_archive_name, json_obj)  # Add the JSON object to the ZIP archive
    zip_buffer.seek(0)
    # Return the zip buffer
    return zip_buffer


def extract_input_zip(content, submission_id=None, problem_id=None, simulation_id=None, step_id=None):
    # Determine the input directory
    input_dir = get_input_dir(submission_id, problem_id, simulation_id, step_id)
    # Create the directory
    os.makedirs(input_dir, exist_ok=True)
    # Extract the zip content
    zip_solution = BytesIO(content)
    with zipfile.ZipFile(zip_solution, 'r') as zip_ref:
        zip_ref.extractall(input_dir)
    # Return the names of the returned files
    return os.listdir(input_dir), input_dir


class DeterministicEvaluationState:
    def __init__(self, prb, es):
        self.prb = prb
        self.es = es
        self.domain = es.domain if es is not None else None
        self.elapsed_time = 0
        self.last_state = None
        self.error_msg = None
        self.plan = BelugaPlan()
        self.final_state = None
        self.final_step = None
        self.goal_reached = False
        self.abrupt_plan_end = False
        self.invalid_plan = False
        self.time_limit_reached = False
        self.step_limit_reached = False

    def get_outcome(self):
        free_racks = self.es._get_free_racks(self.final_state) if self.final_state is not None else 0
        scaled_max_steps = len(self.prb.jigs) * configuration.max_steps
        return SingleSimulationOutcome(plan_construction_time=self.elapsed_time,
                                       error_msg=self.error_msg,
                                       plan=self.plan,
                                       final_state=self.final_state,
                                       final_step=self.final_step,
                                       goal_reached=self.goal_reached,
                                       abrupt_plan_end=self.abrupt_plan_end,
                                       invalid_plan=self.invalid_plan,
                                       time_limit_reached=self.time_limit_reached,
                                       step_limit_reached=(self.final_step==scaled_max_steps-1),
                                       free_racks=free_racks,
                                       prb=self.prb,
                                       alpha=configuration.alpha,
                                       beta=configuration.beta)

    def __repr__(self):
        return self.get_outcome().__repr__()


class CompetitionProcessor:
    def __init__(self, competitor_model_endpoint: str, submission_id: int, resume: bool, reboot_time_limit: float):
        self.competitor_model_endpoint = competitor_model_endpoint
        self.submission_id = submission_id
        self.resume = resume
        self.reboot_time_limit = reboot_time_limit


    def _retrieve_plan(self, eval_state, problem_id, json_file_path):
        # Create an in-memory ZIP file containing the JSON
        logging.debug(f"Problem {problem_id}: compressing JSON file into ZIP format.")
        zip_buffer = zip_json_file(json_file_path, in_archive_name=configuration.problem_file_name)

        # Call the competitor's plan API with the ZIP file and retrieve the solution as a stream
        logging.debug(f"Problem {problem_id}: Sending ZIP to plan API...")
        tstart = time.time()
        solution_response = call_plan_api(
            competitor_model_endpoint=self.competitor_model_endpoint,
            submission_id=self.submission_id,
            problem_id=problem_id,
            zip_file_stream=zip_buffer
        )
        eval_state.elapsed_time += time.time() - tstart
        # print(f"Problem {problem_id}: plan API completed execution")

        # Check for system and communication errors
        if solution_response is None or isinstance(solution_response, dict) and 'error' in solution_response:
            # Wait some time for a possible reboot
            logging.info(f"A communication or resource failure may have happed for {self.submission_id}, problem {problem_id}; waiting for a possible reboot")
            time.sleep(self.reboot_time_limit)
            return None

        # Extract all returned files
        logging.debug(f"Problem {problem_id}: parsing the output of the plan API")
        returned_files, input_dir = extract_input_zip(solution_response, submission_id=self.submission_id, problem_id=problem_id)

        # Check the time limit
        logging.debug(f"Problem {problem_id}: checking time limit")
        if eval_state.elapsed_time > configuration.time_limit:
            eval_state.time_limit_reached = True

        # Atttempt to retrieve the plan
        logging.debug(f"Problem {problem_id}: retrieving the plan")
        if configuration.plan_file_name in returned_files:
            with open(os.path.join(input_dir, configuration.plan_file_name)) as fp:
                json_obj = json.load(fp)
                eval_state.plan = BelugaPlan.from_json_obj(json_obj, eval_state.prb)

        # Return succes flag
        if eval_state.plan is None or eval_state.time_limit_reached:
            return False
        else:
            return True


    def _execute_deterministic_plan(self, eval_state):
        # SIMULATION SETUP ===============================================
        # State fields that cannot be computed based on predicates
        beluga_seq = []
        trailer_location = {}

        # Retrieve the initial state
        state = eval_state.domain.reset()

        # Start plan execution
        scaled_max_steps = len(eval_state.prb.jigs) * configuration.max_steps
        for step in range(scaled_max_steps):
            try:
                # Determine the currently processed flight
                cbeluga = eval_state.es._get_current_beluga(state)
                if len(beluga_seq) == 0 or beluga_seq[-1] != cbeluga:
                    beluga_seq.append(cbeluga)

                # Check whether the goal has been reached
                if eval_state.domain._is_terminal(state):
                    eval_state.goal_reached = True
                    break

                # Check whether the plan has already ended
                if step >= len(eval_state.plan.actions):
                    eval_state.abrupt_plan_end = True
                    break

                # Retrive current action
                ba = eval_state.plan.actions[step]
                # Determine valid actions for the current state
                action = eval_state.es._find_valid_action(ba, state, beluga_seq)

                # Apply the action and move to the next state
                o = eval_state.domain.step(action)
                state = o.observation

                # Update the trailer location
                eval_state.es._update_trailer_location(ba, trailer_location)

            except EvaluationException as e:
                # Build the outcome object
                final_state = eval_state.es._skd_state_to_beluga_state(state=state,
                                                            beluga_seq=beluga_seq,
                                                            trailer_location=trailer_location)
                eval_state.error_msg = e.args[0]
                eval_state.final_state = final_state
                eval_state.final_step = step
                eval_state.invalid_plan = isinstance(e, InvalidActionException)
                return eval_state.get_outcome()

        # The evaluation proceeded normally
        final_state = eval_state.es._skd_state_to_beluga_state(state=state,
                                                    beluga_seq=beluga_seq,
                                                    trailer_location=trailer_location)
        eval_state.final_state = final_state
        eval_state.final_step = step


    def scalability_deterministic_business_logic(self):
        """
        Process logic for the deterministic scalability competition.
        """

        logging.info(f"Starting deterministic competition processing for submission ID {self.submission_id} using model endpoint: {self.competitor_model_endpoint}")

        # Clear the temporary communication directories
        if self.resume == 0:
            clear_tmp_dirs(submission_id=self.submission_id)

        # Synchronous call to the competitor setup API
        logging.info(f"Setup for submission ID {self.submission_id}")
        response = call_setup_api(competitor_model_endpoint=self.competitor_model_endpoint, submission_id=self.submission_id)
        if response is None:
            logging.info(f"Setup API call failed.")
            # Write an error message to the output folder
            eval_state = DeterministicEvaluationState(prb=None, es=None)
            outcome = eval_state.get_outcome()
            outcome.error_msg = 'Setup API call failed'
            write_json_message('outcome', outcome.to_json_obj(),
                               submission_id=self.submission_id)
            # Wait some time for a possible reboot
            logging.info(f"A communication or resource failure may have happed for {self.submission_id}. Waiting for a possible reboot")
            time.sleep(self.reboot_time_limit)
            # write_json_message(name='error', content={'message': 'Setup API call failed'}, submission_id=self.submission_id)
            return # EARLY EXIT

        # Define the list of problems to be processed
        logging.info(f"Loading benchmark problems")
        prb_fnames = glob.glob(f'{configuration.problem_dir}/*.json')
        # prb_fnames = [f for f in os.listdir(configuration.problem_dir) if f.endswith('.json')]
        problems = [(k, f) for k, f in enumerate(sorted(prb_fnames))]

        # Process each problem individually
        outcomes = []
        stored_partial_outcome_json = None
        for problem_id, json_file_path in problems:
            # Check if an outcome is already present
            if self.resume > 0:
                output_dir = get_output_dir(self.submission_id, problem_id)
                outcome_file_path = os.path.join(output_dir, 'outcome.json')
                if os.path.isfile(outcome_file_path):
                    logging.info(f"Resume mode enabled and past outcome deteted for submission {self.submission_id}, problem {problem_id}")
                    if self.resume == 1:
                        # If in normal resume mode, just skip the instace
                        continue
                    elif self.resume == 2:
                        # If in resume (re-evaluation) mode, store the outcome json object
                        with open(outcome_file_path) as fp:
                            stored_partial_outcome_json = json.load(fp)

            logging.info(f"Setup for submission {self.submission_id}, problem {problem_id}")

            # Read problem data
            with open(json_file_path) as fp:
                prb = json.load(fp, cls=BelugaProblemDecoder)

            # Attempt to retrive the stored (partial) outcome. This is only done in resume (re-evaluate) mode
            stored_partial_outcome = None
            if self.resume == 2 and stored_partial_outcome_json:
                stored_partial_outcome = SingleSimulationOutcome.from_json_obj(stored_partial_outcome_json,
                                                                               prb, configuration.alpha, configuration.beta)
            # Build an SKD domain
            domain = SkdPDDLDomain(prb, problem_name=str(problem_id), classic=False)

            # Build a support object
            es = EvaluationSupport(prb, domain)
            es.refresh_cache()

            # Build an object to track the evaluation state
            eval_state = DeterministicEvaluationState(prb, es)

            # Obtain the plan
            logging.info(f"Retrieving a plan for submission {self.submission_id}, problem {problem_id}")
            plan_retrieved = False
            if self.resume == 2 and stored_partial_outcome:
                if stored_partial_outcome.plan and not stored_partial_outcome.time_limit_reached:
                    plan_retrieved = True
                    eval_state.plan = stored_partial_outcome.plan
                    eval_state.elapsed_time = stored_partial_outcome.plan_construction_time
                else:
                    continue # NOTE move to the next instance (there's no valid plan to re-evaluate)
            else:
                plan_retrieved = self._retrieve_plan(eval_state, problem_id, json_file_path)

            # Simulate the plan
            if plan_retrieved is None:
                eval_state.error_msg = 'Plan API call failed (the available time or memory may also have been exceeded'
            elif plan_retrieved:
                logging.info(f"Simulating execution for submission {self.submission_id}, problem {problem_id}")
                self._execute_deterministic_plan(eval_state)

            # Retrieve the final outcome
            outcome = eval_state.get_outcome()

            # Store the final outcome
            outcomes.append(outcome)

            # Write the final outcome
            write_json_message('outcome', outcome.to_json_obj(),
                               submission_id=self.submission_id, problem_id=problem_id)

        # Average all scores
        logging.info(f"Aggregating scores submission {self.submission_id}")
        keys = outcomes[0].score.keys()
        avg_score_dict = {k:np.mean([o.score[k] for o in outcomes]) for k in keys}

        # Write the average scores
        write_json_message('score', avg_score_dict, submission_id=self.submission_id)


    def _setup_problem(self, problem_id, json_file_path):
        # Create an in-memory ZIP file containing the JSON
        logging.debug(f"Problem {problem_id}: compressing JSON file into ZIP format.")
        zip_buffer = zip_json_file(json_file_path, in_archive_name=configuration.problem_file_name)

        # Call the competitor's setup problem API with the ZIP file
        logging.debug(f"Problem {problem_id}: Sending ZIP to setup problem API...")
        response = call_setup_problem_api(
            competitor_model_endpoint=self.competitor_model_endpoint,
            submission_id=self.submission_id,
            problem_id=problem_id,
            zip_file_stream=zip_buffer
        )

        if response is None:
            # Wait some time for a possible reboot
            logging.info(f"A communication or resource failure may have happed for {self.submission_id}, problem {problem_id}. Waiting for a possible reboot")
            time.sleep(self.reboot_time_limit)

        # Return the status of the request
        return response


    def _retrieve_action(self, eval_state, bstate, metadata, problem_id, simulation_id, step_id, past_elapsed_time):
        # zip the current state and metadata
        state_and_metadata = {'state': bstate.to_json_obj(), 'metadata': metadata.to_json_obj()}
        state_and_metadata_zip = zip_json_object(json.dumps(state_and_metadata),
                                                 configuration.state_and_metadata_name)

        # Call the competitor's next_action API with the ZIP file and retrieve the solution as a stream
        logging.debug(f"Request for action {step_id}: Sending ZIP to next action API...")
        tstart = time.time()
        action_response = call_next_action_api(
            competitor_model_endpoint=self.competitor_model_endpoint,
            submission_id=self.submission_id,
            problem_id=problem_id,
            simulation_id=simulation_id,
            action_id=step_id,
            zip_file_stream=state_and_metadata_zip
        )
        eval_state.elapsed_time += time.time() - tstart

        # Check the time limit after the action has been generated
        logging.debug(f"Problem {problem_id}: checking time limit")
        if past_elapsed_time + eval_state.elapsed_time > configuration.time_limit:
            eval_state.time_limit_reached = True

        # Check whether the call was successful
        if action_response is None or (isinstance(action_response, dict) and 'error' in action_response):
            # Wait some time for a possible reboot
            logging.info(f"A communication or resource failure may have happed for {self.submission_id}, problem {problem_id}. Waiting for a possible reboot")
            time.sleep(self.reboot_time_limit)
            return None

        # Extract all returned files
        logging.debug(f"Problem {problem_id}, simulation {simulation_id}, step {step_id}: parsing received action")
        returned_files, input_dir = extract_input_zip(action_response,
                                                      submission_id=self.submission_id, problem_id=problem_id,
                                                      simulation_id=simulation_id, step_id=step_id)

        # Attempt to retrieve the action
        action = None
        if configuration.action_file_name in returned_files:
            with open(os.path.join(input_dir, configuration.action_file_name)) as fp:
                json_obj = json.load(fp)
                action = action_from_json_obj(json_obj, eval_state.prb)

        # Return the action
        return action


    def _run_simulation(self, problem_id, simulation_id, prb, domain, es, past_elapsed_time):
        # Build an object to track the evaluation state
        eval_state = DeterministicEvaluationState(prb, es)

        # Setup a simulation
        logging.debug(f"Setting up simulaiton {simulation_id} for problem {problem_id}")
        response = call_start_simulation_api(competitor_model_endpoint=self.competitor_model_endpoint,
                                             submission_id=self.submission_id,
                                             problem_id=problem_id, simulation_id=simulation_id)

        # If the setup failed, return an outcome
        if response is None:
            # Report an error in the outcome
            eval_state.error_msg = 'simulation setup failed'
            # Wait some time for a possible reboot
            logging.info(f"A communication or resource failure may have happed for {self.submission_id}, problem {problem_id}. Waiting for a possible reboot")
            time.sleep(self.reboot_time_limit)
            return eval_state.get_outcome()

        # Reset the domain state
        state = domain.reset() # This regenerates a PDDL file
        es.refresh_cache() # ...And therefore all translation maps need to be reloaded

        # State fields that cannot be computed based on predicates
        beluga_seq = []
        trailer_location = {}

        # Build a plan dynamically
        eval_state.plan = BelugaPlan()
        scaled_max_steps = len(eval_state.prb.jigs) * configuration.max_steps
        for step_id in range(scaled_max_steps):
            try:
                # Record the step
                eval_state.final_step = step_id

                # Determine the currently processed flight
                cbeluga = es._get_current_beluga(state)
                if len(beluga_seq) == 0 or beluga_seq[-1] != cbeluga:
                    beluga_seq.append(cbeluga)

                # Convert the state
                bstate = es._skd_state_to_beluga_state(state, beluga_seq, trailer_location)
                eval_state.final_state = bstate

                # Check whether the goal has been reached
                if eval_state.domain._is_terminal(state):
                    eval_state.goal_reached = True
                    break

                # First time limit check (used to detect whether the cumulative limit
                # has already been exceeded)
                if past_elapsed_time >= configuration.time_limit:
                    eval_state.time_limit_reached = True
                    return eval_state.get_outcome()

                # Obtain the current metadata
                metadata = ProbabilisticPlanningMetatada(step_id, eval_state.elapsed_time)

                # Retrive current action
                ba = self._retrieve_action(eval_state,
                                           bstate, metadata,
                                           problem_id=problem_id,
                                           simulation_id=simulation_id, step_id=step_id,
                                           past_elapsed_time=past_elapsed_time)

                # Handle timeouts
                if eval_state.time_limit_reached:
                    return eval_state.get_outcome()

                # Handle the case where no action is returned
                if ba is None:
                    eval_state.abrupt_plan_end = True
                    return eval_state.get_outcome()
                else:
                    eval_state.plan.actions.append(ba)

                # Attempt to match the BelugaAction on a valid SKD action
                action = es._find_valid_action(ba, state, beluga_seq)

                # Apply the action and move to the next state
                o = domain.step(action)
                state = o.observation

                # Update the trailer location
                es._update_trailer_location(ba, trailer_location)


            except EvaluationException as e:
                # Build the outcome object
                final_state = eval_state.es._skd_state_to_beluga_state(state=state,
                                                            beluga_seq=beluga_seq,
                                                            trailer_location=trailer_location)
                eval_state.error_msg = e.args[0]
                eval_state.final_state = final_state
                eval_state.final_step = step_id
                eval_state.invalid_plan = isinstance(e, InvalidActionException)
                return eval_state.get_outcome()

        return eval_state.get_outcome()


    def scalability_probabilistic_business_logic(self):
        """
        Process logic for the probabilistic scalability competition.
        """

        logging.info(f"Starting probabilistic competition processing for submission ID {self.submission_id} using model endpoint: {self.competitor_model_endpoint}")

        # Clear the temporary communication directories
        if self.resume == 0:
            clear_tmp_dirs(submission_id=self.submission_id)

        # # Synchronous call to the competitor setup API
        # print(f"Calling setup API for submission ID {self.submission_id}...")
        # response = call_setup_api(competitor_model_endpoint=self.competitor_model_endpoint, submission_id=self.submission_id)
        # if response is None:
        #     # Write an error message to the output folder
        #     print(f"Setup API call failed.")
        #     write_json_message(name='error', content={'message': 'Setup API call failed'}, submission_id=self.submission_id)
        #     return # EARLY EXIT

        # Define the list of problems to be processed
        logging.info(f"Loading benchmark problems for submission {self.submission_id}")
        prb_fnames = glob.glob(f'{configuration.problem_dir_prob}/*.json')
        # prb_fnames = [f for f in os.listdir(configuration.problem_dir) if f.endswith('.json')]
        problems = [(k, f) for k, f in enumerate(sorted(prb_fnames))]

        # Process each problem individually
        outcomes = []
        for problem_id, json_file_path in problems:
            # Check if an outcome is already present
            if self.resume > 0:
                output_dir = get_output_dir(self.submission_id, problem_id)
                if os.path.isfile(os.path.join(output_dir, 'outcome.json')):
                    logging.info(f"Resume mode enabled and past outcome deteted for submission {self.submission_id}, problem {problem_id}")
                    continue

            logging.info(f"Setup for submission {self.submission_id}, problem {problem_id}")

            # Read problem data
            with open(json_file_path) as fp:
                prb = json.load(fp, cls=BelugaProblemDecoder)

            # Build an SKD domain
            domain = SkdSPDDLDomain(prb, problem_name=str(problem_id), classic=False, seed=configuration.seed)

            # Build an support object
            es = EvaluationSupport(prb, domain)

            # Trigger problem setup on the submission side
            status = self._setup_problem(problem_id, json_file_path)
            if status is None:
                # Build an outcome
                eval_state = DeterministicEvaluationState(prb, es)
                eval_state.error_msg = 'problem setup failed'
                write_json_message('outcome', eval_state.get_outcome().to_json_obj(),
                                   submission_id=self.submission_id, problem_id=problem_id)
                continue # NOTE move to the next problem

            # Run simulations
            logging.info(f"Running simulations for submission {self.submission_id}, problem {problem_id}")
            sim_outcomes = []
            total_elapsed_time = 0
            for simulation_id in range(configuration.nsamples):
                # Run a simulation
                logging.debug(f"Simulation {simulation_id} for: {problem_id}")
                sim_outcome = self._run_simulation(problem_id=problem_id, simulation_id=simulation_id,
                                                   prb=prb, domain=domain, es=es, past_elapsed_time=total_elapsed_time)
                # Update elapsed time
                total_elapsed_time += sim_outcome.plan_construction_time
                # Store the outcome in a temporary list
                sim_outcomes.append(sim_outcome)

            # Merge all outcomes
            logging.info(f"Aggregating simulation outcomes for submission {self.submission_id}")
            outcome = MultipleSimulationOutcome(sim_outcomes)
            outcomes.append(outcome)

            # Write the aggregated outcome for this simulation
            write_json_message('outcome', outcome.to_json_obj(),
                               submission_id=self.submission_id, problem_id=problem_id)


        # Average the scores for all problems
        keys = outcomes[0].individual_outcomes[0].score.keys()
        avg_score_dict = {k:np.mean([o._avg_score_dict()[k] for o in outcomes]) for k in keys}

        # Write the average scores
        write_json_message('score', avg_score_dict, submission_id=self.submission_id)
