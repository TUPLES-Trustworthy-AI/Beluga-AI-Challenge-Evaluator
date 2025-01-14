# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Install dependencies for building pynng
RUN apt-get update && apt-get install git gcc make -y

WORKDIR /app

# Copy the requirements file (you can update it using `pip freeze > requirements.txt`)
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Install the tool dependencies
RUN pip install --no-cache-dir -r src/tools/requirements.txt

# Set the Python path to include both `src` and `.` (the root) to handle namespace packages
ENV PYTHONPATH="/app:/app/src:/app/web_service"

# Set the working directory to the src folder
WORKDIR /app/web_service

# Expose the port on which the app will run
EXPOSE 80

# Command to run the app
CMD ["python", "-u", "webservice.py"]
