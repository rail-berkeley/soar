#!/bin/bash

# Allow overriding variables from the command line
SAVE_DIR="${SAVE_DIR:-$HOME/soar_data_numpy_source}"
REQUIRED_SPACE_GB="${REQUIRED_SPACE_GB:-30}"

# File dir
REMOTE_FILE_DIR="https://rail.eecs.berkeley.edu/datasets/soar_release/soar-dataset-numpy.zip"

# Function to check if enough disk space is available
check_disk_space() {
    local available_space
    available_space=$(df --output=avail -BG . | tail -1 | tr -d 'G')
    if [ "$available_space" -lt "$REQUIRED_SPACE_GB" ]; then
        echo "Warning: You need at least $REQUIRED_SPACE_GB GB of free space to proceed."
        read -p "Do you want to continue anyway? (y/n) " REPLY
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting download."
            exit 1
        fi
    fi
}

# Check disk space
check_disk_space

# Inform saving directory
echo "Saving files to $SAVE_DIR"

# Download the dataset
mkdir -p "$SAVE_DIR"
wget -O "$SAVE_DIR/soar-dataset-numpy.zip" "$REMOTE_FILE_DIR"

# Unzip the dataset
unzip "$SAVE_DIR/soar-dataset-numpy.zip" -d "$SAVE_DIR"

