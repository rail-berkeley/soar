# This script will take a long time to run, and the cached result in already saved in urls.txt
# You shouldn't need to rerun this

# List files, filter by patterns, remove duplicates, and save to urls.txt
BASE_URL="https://rail.eecs.berkeley.edu/datasets/soar_release/1.0.0/"

echo "Fetching all datafile URLs..."
wget --spider -r -nd -np $BASE_URL 2>&1 | grep '^--' | awk '{ print $3 }' | grep -E '\.json$|tfrecord' | sort | uniq > urls.txt
echo "Finished fetching URLs."
