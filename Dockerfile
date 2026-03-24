# 1. Start from a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# Accept RUN_ID as an argument
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

# 3. Efficient layering: Copy only the requirements file first
# This allows Docker to cache the installed dependencies as long as requirements.txt doesn't change
COPY requirements.txt .

# 4. Install dependencies
# Using --no-cache-dir keeps the final image size as small as possible
RUN pip install -r requirements.txt

# 5. Copy the rest of the application code into the container
COPY . .

# MLflow Tracking Configuration
ARG MLFLOW_TRACKING_URI
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ARG MLFLOW_TRACKING_USERNAME
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
ARG MLFLOW_TRACKING_PASSWORD
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}

# Download the model from DagsHub/MLflow
RUN python download_model.py

# 6. Define the default command to run when the container starts
CMD ["python", "train.py"]