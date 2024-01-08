# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster
# Set the working directory in the container to /app
WORKDIR /app

COPY ./ /app

# Copy the requirements file into the container
# COPY ./requirements.txt /app/requirements.txt

# Install gcc and other dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip  -r requirements.txt
