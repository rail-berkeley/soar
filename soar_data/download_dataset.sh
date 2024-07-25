#!/bin/bash

# Define variables
BASE_URL="https://rail.eecs.berkeley.edu/datasets/soar_release/1.0.0/"
SAVE_DIR="~/soar_data"
REQUIRED_SPACE_GB=140
URL_FILE="soar_data/urls.txt"

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

# Check if the url file exists
if [ ! -f $URL_FILE ]; then
    echo "Error: URLs file not found. Please run from root directory of the repo."
    echo "URL file should be found at soar_data/urls.txt"
    echo "Aborting download."
    exit 1
fi

# Inform saving directory
echo "Saving files to $SAVE_DIR"

# Count the number of files
TOTAL_FILES=$(wc -l < $URL_FILE)
CURRENT_FILE=0

# Function to print the progress bar
print_progress() {
    local PROGRESS=$(( ($CURRENT_FILE * 100) / $TOTAL_FILES ))
    local FILLED=$(( $PROGRESS / 2 ))
    local EMPTY=$(( 50 - $FILLED ))
    printf "\rDownloading files: ["
    printf "%0.s#" $(seq 1 $FILLED)
    printf "%0.s " $(seq 1 $EMPTY)
    printf "] %d%%" $PROGRESS
}

# Function to download using wget without parallel
download_without_parallel() {
    while IFS= read -r url; do
        wget -P "$SAVE_DIR" "$url"
        ((CURRENT_FILE++))
        print_progress
    done < $URL_FILE
    echo
}

download_with_parallel() {
    cat $URL_FILE | parallel -j 4 wget -P "$SAVE_DIR" {}
}

# Check if parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU parallel is not installed."
    read -p "Do you want to install GNU parallel? (y/n) " REPLY
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Try to install parallel
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y parallel
        elif command -v yum &> /dev/null; then
            sudo yum install -y parallel
        elif command -v brew &> /dev/null; then
            brew install parallel
        else
            echo "Package manager not found. Please install GNU parallel manually."
            download_without_parallel
            exit 1
        fi
    else
        echo "Downloading files without parallelism..."
        download_without_parallel
        exit 0
    fi
fi

download_with_parallel

# Initialize the progress bar
CURRENT_FILE=0
print_progress

# Monitor the progress of parallel downloads
while IFS= read -r url; do
    if [ -f "$SAVE_DIR/$(basename $url)" ]; then
        ((CURRENT_FILE++))
        print_progress
    fi
done < $URL_FILE
echo
