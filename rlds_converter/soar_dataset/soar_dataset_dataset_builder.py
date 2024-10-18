import glob
import os

import cv2
import numpy as np
import tensorflow_datasets as tfds
from absl import logging
from dataset_builder import MultiThreadedDatasetBuilder

IMAGE_SIZE = (256, 256)
TRAIN_PROPORTION = 1.0


# Function to read frames from a video and store them as a numpy array
def video_to_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    frames = []
    while True:
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Convert BGR to RGB as OpenCV uses BGR by default
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Append the RGB frame to the frames list
        frames.append(frame_rgb)

    # Close the video file
    cap.release()

    # Convert the list of frames to a numpy array
    frames_array = np.stack(frames, axis=0)

    return frames_array


def process_images(path):  # processes images at a trajectory level
    video_dir = os.path.join(path, "trajectory.mp4")
    assert os.path.exists(video_dir), f"Video file {video_dir} does not exist"
    frames = video_to_frames(video_dir)
    return frames


def process_goals(path):  # processes images at a trajectory level
    video_dir = os.path.join(path, "goals.mp4")
    frames = video_to_frames(video_dir)
    return frames


def process_state(path):
    eef = os.path.join(path, "eef_poses.npy")
    return np.load(eef)


def process_actions(path):
    actions_path = os.path.join(path, "actions.npy")
    return list(np.load(actions_path))


def process_lang(path):
    fp = os.path.join(path, "language_task.txt")
    text = ""  # empty string is a placeholder for missing text
    if os.path.exists(fp):
        with open(fp, "r") as f:
            text = f.readline().strip()

    return text


def read_txt(path, not_exist_ok=False):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.readline().strip()
    else:
        if not_exist_ok:
            return ""
        else:
            raise ValueError(f"File {path} does not exist")


def process_success(path):
    success_text = read_txt(os.path.join(path, "success.txt"))
    assert success_text in ("True", "False")
    return success_text == "True"


class SOARDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for soar dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Please see official release"

    NUM_WORKERS = 16
    CHUNKSIZE = 1000

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_0": tfds.features.Image(
                                        shape=IMAGE_SIZE + (3,),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation (fixed position).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot end effector state, consists of [3x XYZ, 3x roll-pitch-yaw, 1x gripper]",
                                    ),
                                }
                            ),
                            "goal": tfds.features.Image(
                                shape=IMAGE_SIZE + (3,),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                                doc="Goal image. Generated by SuSIE.",
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [3x XYZ delta, 3x roll-pitch-yaw delta, 1x gripper absolute].",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                            "has_language": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if language exists in observation, otherwise empty string.",
                            ),
                            "task_list": tfds.features.Text(
                                doc="List of language task that is available for the scene."
                            ),
                            "object_list": tfds.features.Text(
                                doc="List of object in the scene."
                            ),
                            "time": tfds.features.Text(
                                doc="Time the episode is recorded."
                            ),
                            "robot_id": tfds.features.Text(doc="Robot ID."),
                            "success": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True if the episode is deemed successful by VLM, otherwise False.",
                            ),
                        }
                    ),
                }
            )
        )

    @classmethod
    def _process_example(cls, example_input):
        """Process a single example."""
        path = example_input

        out = dict()

        out["images"] = process_images(path)
        out["goals"] = process_goals(path)
        out["state"] = process_state(path)
        out["actions"] = process_actions(path)
        out["lang"] = process_lang(path)

        assert len(out["actions"]) == len(out["state"]) == len(out["images"]), (
            path,
            len(out["actions"]),
            len(out["state"]),
            len(out["images"]),
        )

        instruction = out["lang"]
        assert (
            instruction != ""
        ), f"Empty instruction at {path}"  # all SOAR data have language

        # assemble episode
        episode = []
        episode_metadata = {
            "file_path": path,
            "has_language": bool(instruction),
            "task_list": read_txt(
                os.path.join(path, "task_list.txt"),
                not_exist_ok=True,
            ),
            "object_list": read_txt(
                os.path.join(path, "object_list.txt"), not_exist_ok=True
            ),
            "time": read_txt(
                os.path.join(path, "time.txt"),
                not_exist_ok=True,
            ),
            "robot_id": read_txt(
                os.path.join(path, "robot_id.txt"),
                not_exist_ok=True,
            ),
            "success": process_success(path),
        }

        try:

            for i in range(len(out["actions"])):
                observation = {
                    "state": out["state"][i].astype(np.float32),
                    "image_0": out["images"][i],
                }

                episode.append(
                    {
                        "observation": observation,
                        "goal": out["goals"][i],
                        "action": out["actions"][i].astype(np.float32),
                        "is_first": i == 0,
                        "is_last": i == (len(out["actions"]) - 1),
                        "language_instruction": instruction,
                    }
                )

        except IndexError:
            print(f"IndexError at {path}")
            for k, v in out.items():
                print(k, len(v))

        # create output data sample
        sample = {"steps": episode, "episode_metadata": episode_metadata}

        # use episode path as key
        return path, sample

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # this function assumes the data structure is organized as follows:
        # manual_dir/robot_id/scene_id/policy_type/date/success/trajectory_{i}/*
        # manual_dir/robot_id/scene_id/policy_type/date/failure/trajectory_{i}/*
        DEPTH = 4  # the number of subdirectories before success/failure dirs
        paths = glob.glob(os.path.join(dl_manager.manual_dir, *("*" * DEPTH)))

        success_inputs, failure_inputs = [], []

        for path in paths:
            search_success = os.path.join(path, "success", "traj*")
            search_failure = os.path.join(path, "failure", "traj*")
            all_traj_success = glob.glob(search_success)
            all_traj_failure = glob.glob(search_failure)
            if not all_traj_success:
                print(f"no trajs found in {search_success}")
                continue
            if not all_traj_failure:
                print(f"no trajs found in {search_failure}")
                continue

            success_inputs += all_traj_success
            failure_inputs += all_traj_failure

        logging.info(
            "Converting %d success and %d failure files.",
            len(success_inputs),
            len(failure_inputs),
        )
        return {
            "success": iter(success_inputs),
            "failure": iter(failure_inputs),
        }
