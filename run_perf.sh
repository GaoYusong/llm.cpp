#!/bin/bash

# Get the current time
current_time=$(date +"%Y%m%d_%H%M%S")

# Get the latest git commit hash
latest_commit=$(git rev-parse --short HEAD)

# Create the perf directory if it doesn't exist
mkdir -p perf

# Define the output file name
output_file="perf/${current_time}_${latest_commit}.txt"

# Run ./test_gpt2 and save the output to both the file and the console
./test_gpt2 | tee "$output_file"

# Print a completion message
echo "Output has been saved to $output_file"

