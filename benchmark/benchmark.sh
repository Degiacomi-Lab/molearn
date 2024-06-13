#!/bin/bash


# Name of the conda environment
ENV_NAME="molearn"

# Initialize Conda
source /home/pghw87/miniconda3/etc/profile.d/conda.sh

# Activate the environment
conda activate "$ENV_NAME"

LOG_DIR="./Log"

# Define the base path where Python scripts are located
PYTHON_SCRIPTS_PATH="/home/pghw87/Documents/molearn/Eng-asal/molearn/examples"

# Define an array of Python training scripts
PYTHON_SCRIPTS=("siren_basic.py" "siren_basic_3x1024.py" "siren_basic_4x512.py" "siren_basic_baa.py" "siren_basic_aba.py" "siren_basic_aab.py") 
# The first script is different than previously trained network in soft_nb



# Loop through each script in the array
for script in "${PYTHON_SCRIPTS[@]}"; do
    TIMESTAMP=$(date +%d-%H%M%S)
    LOG_FILE="${LOG_DIR}/${script%.py}_benchmark_${TIMESTAMP}.log"

    echo "Starting training for $script: $(date)" > "$LOG_FILE"
    start_time=$(date +%s) # Record start time

    # Run the Python script with CUDA support, assuming CUDA set up is available
    CUDA_VISIBLE_DEVICES=0 python "${PYTHON_SCRIPTS_PATH}/${script}" >> "$LOG_FILE" 2>&1

    end_time=$(date +%s) # Record end time
    duration=$((end_time - start_time)) # Calculate duration

    echo "Training for $script completed in $duration seconds: $(date)" >> "$LOG_FILE"
done

echo "All training sessions completed: $(date)"
