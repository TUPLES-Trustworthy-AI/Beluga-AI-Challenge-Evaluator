import logging
from io import BytesIO
import requests

# Configure the logging module to log messages with time, level, and message format.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.DEBUG)

# 1. Function to call the `setup` endpoint
def call_setup_api(competitor_model_endpoint: str, submission_id: int):
    """
    Calls the setup API to configure the competitor model using the given submission ID.

    Args:
        competitor_model_endpoint (str): The URL of the setup API endpoint.
        submission_id (int): The ID of the current submission.

    Returns:
        dict: The JSON response from the setup API if the call is successful.
    """
    logging.debug(f"Calling setup API for submission_id: {submission_id}")

    url = f"{competitor_model_endpoint}/setup"
    params = {"submission_id": submission_id}

    try:
        # Send a POST request to the setup API with the submission ID as a query parameter.
        response = requests.post(url=url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors.

        logging.debug(f"Setup API call successful for submission_id: {submission_id}")
        return response.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"An HTTP error occurred during the setup API call: {e}")
        return None
    except Exception as err:
        logging.error(f"An unexpected error occurred during the setup API call: {err}")
        return None


# 2. Function to call the `plan` endpoint
def call_plan_api(competitor_model_endpoint: str, submission_id: int, problem_id: int, zip_file_stream: BytesIO):
    """
    Calls the plan API, sending a ZIP file with the problem data to get a solution.

    Args:
        competitor_model_endpoint (str): The API endpoint for the competitor model.
        submission_id (int): The submission ID.
        problem_id (int): The problem ID.
        zip_file_stream (BytesIO): A ZIP file stream containing the problem data.

    Returns:
        object: The API response as a ZIP file if successful.
    """
    logging.debug(f"Calling plan API for problem_id: {problem_id}, submission_id: {submission_id}")

    url = f"{competitor_model_endpoint}/plan?submission_id={submission_id}&problem_id={problem_id}"
    headers = {'Content-Type': 'application/octet-stream'}

    try:
        # Send a POST request to the plan API with the ZIP file.
        response = requests.post(url, headers=headers, data=zip_file_stream.getvalue())

        if response.status_code == 200:
            # Check if the response contains a ZIP file.
            if response.headers.get('Content-Type') == 'application/octet-stream':
                logging.debug(f"Plan API returned a ZIP file for problem_id: {problem_id}")
                return response.content
            else:
                logging.warning(f"Plan API did not return the expected content-type for problem_id: {problem_id}")
                return {"error": "Missing content-type application/octet-stream in the plan API response"}
        else:
            logging.warning(f"Plan API call failed with status code: {response.status_code}")
            return {"error": response.json(), "status_code": response.status_code}

    except Exception as e:
        logging.error(f"An error occurred during the plan API call: {e}")
        return {"error": str(e)}


# 3. Function to call the `setup_problem` endpoint
def call_setup_problem_api(competitor_model_endpoint: str, submission_id: int, problem_id: int,
                           zip_file_stream: BytesIO):
    """
    Calls the setup_problem API, sending a ZIP file to configure the problem.

    Args:
        competitor_model_endpoint (str): The API endpoint for the competitor model.
        submission_id (int): The submission ID.
        problem_id (int): The problem ID.
        zip_file_stream (BytesIO): A ZIP file stream containing the problem configuration.

    Returns:
        dict: The JSON response from the setup_problem API if successful.
    """
    logging.debug(f"Calling setup_problem API for submission_id: {submission_id}, problem_id: {problem_id}")

    url = f"{competitor_model_endpoint}/setup_problem?submission_id={submission_id}&problem_id={problem_id}"
    headers = {'Content-Type': 'application/octet-stream'}

    try:
        # Send a POST request to the setup_problem API with the ZIP file.
        response = requests.post(url, headers=headers, data=zip_file_stream.getvalue())
        response.raise_for_status()  # Raise an exception for HTTP errors.

        logging.debug(f"Setup_problem API call was successful for problem_id: {problem_id}")
        return response.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"An HTTP error occurred during the setup_problem API call: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during the setup_problem API call: {e}")
        return {"error": str(e)}


# 4. Function to call the `start_simulation` endpoint
def call_start_simulation_api(competitor_model_endpoint: str, submission_id: int, problem_id: int, simulation_id: int):
    """
    Calls the start_simulation API to initiate a simulation.

    Args:
        competitor_model_endpoint (str): The URL of the start_simulation endpoint.
        submission_id (int): The submission ID.
        problem_id (int): The problem ID.
        simulation_id (int): The simulation ID.

    Returns:
        dict: The JSON response from the start_simulation API if successful.
    """
    logging.debug(
        f"Calling start_simulation API for submission_id: {submission_id}, problem_id: {problem_id}, simulation_id: {simulation_id}")

    url = f"{competitor_model_endpoint}/start_simulation?submission_id={submission_id}&problem_id={problem_id}&simulation_id={simulation_id}"

    try:
        # Send a POST request to the start_simulation API.
        response = requests.post(url=url)
        response.raise_for_status()  # Raise an exception for HTTP errors.

        logging.debug(f"Start_simulation API call was successful for simulation_id: {simulation_id}")
        return response.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"An HTTP error occurred during the start_simulation API call: {e}")
        return None
    except Exception as err:
        logging.error(f"An unexpected error occurred during the start_simulation API call: {err}")
        return None


# 5. Function to call the `next_action` endpoint
def call_next_action_api(competitor_model_endpoint: str, submission_id: int, problem_id: int, simulation_id: int,
                         action_id: int, zip_file_stream: BytesIO):
    """
    Calls the next_action API, sending the action data for a specific simulation step.

    Args:
        competitor_model_endpoint (str): The API endpoint for the competitor model.
        submission_id (int): The submission ID.
        problem_id (int): The problem ID.
        simulation_id (int): The simulation ID.
        action_id (int): The action ID.
        zip_file_stream (BytesIO): A ZIP file stream containing the action data.

    Returns:
        object: The API response as a ZIP file if successful.
    """
    logging.debug(
        f"Calling next_action API for submission_id: {submission_id}, problem_id: {problem_id}, simulation_id: {simulation_id}, action_id: {action_id}")

    url = f"{competitor_model_endpoint}/next_action?submission_id={submission_id}&problem_id={problem_id}&simulation_id={simulation_id}&action_id={action_id}"
    headers = {'Content-Type': 'application/octet-stream'}

    try:
        # Send a POST request to the next_action API with the ZIP file.
        response = requests.post(url, headers=headers, data=zip_file_stream.getvalue())

        if response.status_code == 200:
            if response.headers.get('Content-Type') == 'application/octet-stream':
                logging.debug(f"Next_action API returned a ZIP file for action_id: {action_id}")
                return response.content
            else:
                logging.warning(f"Next_action API did not return the expected content-type for action_id: {action_id}")
                return {"error": "Missing content-type application/octet-stream in the next_action API response"}
        else:
            logging.warning(f"Next_action API call failed with status code: {response.status_code}")
            return {"error": response.json(), "status_code": response.status_code}

    except Exception as e:
        logging.error(f"An error occurred during the next_action API call for action_id: {action_id}, Error: {e}")
        return {"error": str(e)}
