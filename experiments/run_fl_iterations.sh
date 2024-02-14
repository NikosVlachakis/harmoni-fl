#!/bin/bash

# Function to handle script termination
cleanup() {
    echo "Terminating the script. Cleaning up..."
    docker-compose down
    exit 1 # Exit the script with a non-zero status
}

# Function to wait for the federated learning process to finish
waitForFLCompletion() {
    echo "Waiting for the federated learning process to finish..."

    # Monitor all containers part of the docker-compose setup for the "FL finished" message
    # This command combines logs from all containers and stops waiting when the message is found
    docker-compose logs -f 2>&1 | grep -m 1 "FL finished"
    echo "Federated learning process has finished."
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Check if an argument was provided (number of iterations)
if [ $# -eq 0 ]; then
    echo "Please provide the number of iterations as an argument."
    exit 1
fi

# Get the number of iterations from the script's first argument
NUM_ITERATIONS=$1

echo "Starting the federated learning process for $NUM_ITERATIONS iterations."

for ((i=1; i<=NUM_ITERATIONS; i++))
do
    echo "Iteration $i of $NUM_ITERATIONS"
    
    # Stop and remove the Docker Compose services if they are already running
    docker-compose down
    
    # Start the Docker Compose services in detached mode
    docker-compose up client1 client2 client3 client4 server
    
    # Call the function to wait for the federated learning process to finish
    waitForFLCompletion
    
    echo "Iteration $i completed."
    
    # Here you can add additional commands to collect logs, metrics, or results
    # from each iteration as needed for analysis.
done

echo "All iterations completed."
