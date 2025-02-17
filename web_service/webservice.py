import time
import asyncio

import requests
from concurrent.futures import ProcessPoolExecutor
from enum import Enum

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

import logging

from businesslogic import CompetitionProcessor

import tomllib
import os

# Class to model configuration options
class Configuration:
    def __init__(self):
        self.send_to_orchestrator = False
        self.resume = False

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
        if 'send_to_orchestrator' in conf_data['infrastructure']:
            res.send_to_orchestrator = conf_data['infrastructure']['send_to_orchestrator']
        if 'resume' in conf_data['infrastructure']:
            res.resume = conf_data['infrastructure']['resume']
    except Exception:
        print('[INFRASTRUCTURE] Cannot read configuration file, using default configuration options')

    return res

# Read configuration options
configuration = conf_from_file()

# Build the FastAPI object
app = FastAPI()

# Create a global process pool for heavy tasks
executor = ProcessPoolExecutor()

# Define the CompetitionType enum to handle the allowed values
class CompetitionType(str, Enum):
    SCALABILITY_DETERMINISTIC = "scalability_deterministic"
    SCALABILITY_PROBABILISTIC = "scalability_probabilistic"
    EXPLAINABILITY = "explainability"

@app.post("/run", status_code=200)
async def run(submission_id: int = Query(...),
              competition_type: CompetitionType = Query(...),
              optit_endpoint: str = Query(...),
              competitor_model_endpoint: str = Query(...)):
    """
    Handles incoming POST requests for new submission processing.

    Args:
    - submission_id: The unique identifier for this submission (from query string).
    - competition_type: The type of competition. Must be one of the values from the CompetitionType enum (from query string).
    - optit_endpoint: The endpoint where the solution will be posted once processed (from query string).
    - competitor_model_endpoint: The endpoint for the competitor's model (from query string).

    Returns:
    - A JSON response indicating that the request has been accepted and is processing in the background.
    """

    # The competition_type is already validated by the Enum, no need to check again

    # Immediately trigger the processing task in the background without blocking the request
    asyncio.create_task(
        handle_request(submission_id=submission_id, competition_type=competition_type.value,
                       optit_endpoint=optit_endpoint, competitor_model_endpoint=competitor_model_endpoint)
    )

    # Return a JSON response indicating that the request has been accepted for background processing
    return JSONResponse(content={"message": "Request accepted, processing in the background."})

async def handle_request(submission_id: int, competition_type: str, optit_endpoint: str, competitor_model_endpoint: str):
    try:
        # Get the event loop to run the process in a background executor (separate CPU-bound thread)
        loop = asyncio.get_event_loop()

        # Execute the process_and_send_zip function in a separate process using ProcessPoolExecutor.
        # This is used for CPU-bound tasks to prevent blocking the main thread.
        await loop.run_in_executor(
            executor,  # Executor (assumed to be defined elsewhere) for managing parallel processes
            process,  # Function to execute
            submission_id,
            competition_type,
            optit_endpoint,
            competitor_model_endpoint
        )

    except Exception as e:
        # Handle any errors that occur during processing and log them
        print(f"Error in sub-process: {e}")


def process(submission_id: int, competition_type: CompetitionType, optit_endpoint: str, competitor_model_endpoint: str):
    """
    Process the submission based on the competition type.

    Args:
    - submission_id: The unique identifier for the submission.
    - competition_type: The type of competition. Determines which business logic to call (using CompetitionType enum).
    - optit_endpoint: The endpoint to send the output to.
    - competitor_model_endpoint: The endpoint for the competitor's model, passed to the business logic functions.
    """

    # Print a message to indicate that computation is starting (for logging/debugging purposes)
    print(f"Starting the process for submission ID {submission_id}...")  # "I'm computing.."

    # Instantiating the processor
    processor = CompetitionProcessor(competitor_model_endpoint, submission_id, configuration.resume)

    #Process the submission based on competition type using the CompetitionType enum
    if competition_type == CompetitionType.SCALABILITY_DETERMINISTIC:
        print(f"Starting the deterministic process for submission ID {submission_id}...")
        # Call deterministic business logic
        processor.scalability_deterministic_business_logic()
    elif competition_type == CompetitionType.SCALABILITY_PROBABILISTIC:
        print(f"Starting the deterministic process for submission ID {submission_id}...")
        # Call probabilistic business logic
        processor.scalability_probabilistic_business_logic()
    else:
        # If the competition type is not recognized, raise an error (though Enum should ensure validity)
        raise ValueError(f"Invalid competition type: {competition_type}")

    # Send the generated output to the specified endpoint, including temp_path if required
    if configuration.send_to_orchestrator:
        send_output(submission_id=submission_id, optit_endpoint=optit_endpoint)

    # Print a message to indicate that processing is complete (for logging/debugging purposes)
    print(f"Processing complete for submission ID {submission_id}.")

def send_output(submission_id, optit_endpoint, retries=5, delay=2):
     # Log message indicating the start of packing output files into a zip
    print("Sending output to OPTIT Model orchestrator... ")

    # Set the request headers to specify the content type as binary
    headers = {}

    # Construct the URL for sending the result, including the execution_id as a query parameter
    url = f"{optit_endpoint}/result?submission_id={submission_id}"

    # Attempt to send the zip file to the solution endpoint, with retry logic
    for attempt in range(retries):
        try:
            # Send a POST request with the zip file data
            response = requests.post(url, headers=headers)

            # INSERT HERE OUTPUT TO OPTIT_MODEL_ORCHESTRATOR

            # Check if the response indicates success
            if response.status_code == 200:
                print(f"Result successfully sent to {optit_endpoint}")
                return  # Exit the function after a successful response
            else:
                # Log the failure status code and current attempt number
                print(f"Failed to send result. Status code: {response.status_code}. Attempt {attempt + 1} of {retries}")

        except requests.exceptions.RequestException as e:
            # Log any request exceptions that occur and the current attempt number
            print(f"Request failed due to: {e}. Attempt {attempt + 1} of {retries}")

        # If there are more attempts remaining, wait before retrying
        if attempt < retries - 1:
            time.sleep(delay)  # Wait for the specified delay
            print(f"Waiting {delay} seconds before retrying...")
            delay += delay  # Exponentially increase the delay for subsequent attempts

    # Log a message if all retry attempts fail
    print("Failed to send result after multiple attempts.")


# Run the service on port 80
if __name__ == "__main__":
    # Start the ASGI server using Uvicorn to serve the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=80)  # Host the application on all available IP addresses at port 80
