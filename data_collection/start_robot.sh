#!/bin/bash

# Check if the first argument is provided
if [ -z "$1" ]; then
  echo "No argument provided. Please provide a number between 0 and 4."
  exit 1
fi

# Check if the first argument is a valid number between 0 and 4
if ! [[ "$1" =~ ^[0-4]$ ]]; then
  echo "Invalid argument. Please provide a number between 0 and 4."
  exit 1
fi

# Extract the first argument and the rest of the arguments
n="$1"
shift
additional_args="$@"

# If reset flag is true, execute the reset commands
# echo "Resetting ports 7000 and 6000"
# lsof -ti:7000 | xargs kill -9
# lsof -ti:6000 | xargs kill -9

# Execute the ssh port forwarding script
echo "Executing: bash ssh_port_forward.sh"
bash ssh_port_forward.sh

# Construct the command based on the first argument
config_dir="config/berkeley_robot_$n"
command="python orchestrator/robot/main.py --config_dir $config_dir $additional_args"

# Execute the Python command
echo "Executing: $command"
$command
