# Data RLDS conversion code

This directory contains the code that converts the raw robot data logged using the `data_collection` dir and converts it into
the [RLDS format](https://github.com/google-research/rlds), which is a specification on top of the [TFDS](https://www.tensorflow.org/datasets) (TensorFlow datasets) format, which is for the most part built on top of the TFRecord format.

This is sourced from [Zhiyuan Zhou's implementation](https://github.com/zhouzypaul/dlimp), which heavily inherits Kevin Black's [dlimp](https://github.com/kvablack/dlimp) library.


## Usage
Install the requirements with
```bash
pip install -r requirements.txt
pip install -e .
```

To build the SOAR dataset
```bash
cd soar_dataset
CUDA_VISIBLE_DEVICES="" tfds build --manual_dir <path_to_raw_data>
```
You can modify settings in side the `soar_dataset/soar_dataset_dataset_builder.py` file (e.g., `NUM_WORKERS` and `CHUNKSIZE`).

This data builder assumes your raw data is organized into the following structure:
```
manual_dir/robot_id/scene_id/policy_type/date/success/trajectory_{i}/*
manual_dir/robot_id/scene_id/policy_type/date/failure/trajectory_{i}/*
```
Each `trajectory_{i}` directory will contain the following files, as logged by the code in the `data_collection` dir
- actions.npy
- eef_poses.npy
- language_task.txt
- robot_id.txt
- task_list.txt
- trajectory.mp4
- combined.mp4
- goals.mp4
- object_list.txt
- success.txt
- time.txt

The RLDS dataset will be automatically saved under `~/tensorflow_datasets/soar_dataset/`
