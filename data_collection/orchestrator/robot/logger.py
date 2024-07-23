import os

import cv2
import numpy as np
import os
import subprocess
import json

class Logger:
    def __init__(self, config):
        self.robot_id = str(config["general_params"]["robot_id"])
        self.video_save_path = config["general_params"]["video_save_path"]
        # make sure video_save_path exists
        if not os.path.exists(self.video_save_path):
            os.makedirs(self.video_save_path)

        # Find the highest index trajectory already saved
        # in the save folder, and start counting trajectories
        # with one above that number
        indices = [
            int(name[4:])
            for name in os.listdir(self.video_save_path)
            if os.path.isdir(os.path.join(self.video_save_path, name))
        ]
        self.traj_idx = max(indices) + 1 if len(indices) != 0 else 0

        # Get image sizes from the config
        self.obs_image_size = config["gc_policy_params"]["image_size"]
        self.goal_image_size = config["subgoal_predictor_params"]["image_size"]

        # Initialize data structures for things we are logging
        self.obs_images = []
        self.goal_images = []
        self.actions = []
        self.poses = []

        # We will also log the scene information for each trajectory
        self.object_list = config["task_definition_params"]["object_list"]
        self.task_list = config["task_definition_params"]["task_list"]

    def log_obs(self, image: np.ndarray):
        assert image.shape == (
            self.obs_image_size,
            self.obs_image_size,
            3,
        ), "Cannot log incorrectly shaped obs image"
        self.obs_images.append(image)

    def log_goal(self, image: np.ndarray):
        assert image.shape == (
            self.goal_image_size,
            self.goal_image_size,
            3,
        ), "Cannot log incorrectly shaped goal image"
        self.goal_images.append(image)

    def log_action(self, action: np.ndarray):
        assert action.shape == (7,), "Action should have 7 dimensions"
        self.actions.append(action)

    def log_pose(self, pose: np.ndarray):
        """
        This method logs the pose of the robot before the action is taken
        """
        assert pose.shape == (7,), "Robot pose should have 7 dimensions"
        self.poses.append(pose)
    
    def reset(self):
        self.obs_images = []
        self.goal_images = []
        self.actions = []
        self.poses = []

    def flush_trajectory(self, commanded_task: str, success: bool, log_combined: bool = True):
        subdir_path = os.path.join(self.video_save_path, "traj" + str(self.traj_idx))
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

        # Log the language task
        with open(os.path.join(subdir_path, "language_task.txt"), "w") as f:
            f.write(commanded_task)

        # Log the success information
        with open(os.path.join(subdir_path, "success.txt"), "w") as f:
            f.write(str(success))

        # Log the actions
        np.save(os.path.join(subdir_path, "actions.npy"), np.array(self.actions))

        # Log the robot poses
        np.save(os.path.join(subdir_path, "eef_poses.npy"), np.array(self.poses))

        # Log the observation video
        size = (self.obs_image_size, self.obs_image_size)
        out = cv2.VideoWriter(
            os.path.join(subdir_path, "trajectory.mp4"),
            cv2.VideoWriter_fourcc(*"DIVX"),
            15,
            size,
        )
        for i in range(len(self.obs_images)):
            rgb_img = cv2.cvtColor(self.obs_images[i], cv2.COLOR_RGB2BGR)
            out.write(rgb_img)
        out.release()

        # Log the goals video
        size = (self.goal_image_size, self.goal_image_size)
        out = cv2.VideoWriter(
            os.path.join(subdir_path, "goals.mp4"),
            cv2.VideoWriter_fourcc(*"DIVX"),
            15,
            size,
        )
        for i in range(len(self.goal_images)):
            rgb_img = cv2.cvtColor(self.goal_images[i], cv2.COLOR_RGB2BGR)
            out.write(rgb_img)
        out.release()

        # Log the combined image
        if log_combined:
            assert (
                self.obs_image_size == self.goal_image_size
            ), "To log combined video obs and goal images must be the same size"
            assert len(self.obs_images) == len(
                self.goal_images
            ), "To log combined video there must be equal number of obs and goal images"
            size = (self.obs_image_size + self.goal_image_size, self.obs_image_size)
            out = cv2.VideoWriter(
                os.path.join(subdir_path, "combined.mp4"),
                cv2.VideoWriter_fourcc(*"DIVX"),
                15,
                size,
            )
            for i in range(len(self.goal_images)):
                combined_image = np.concatenate(
                    [self.obs_images[i], self.goal_images[i]], axis=1
                )
                rgb_img = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
                out.write(rgb_img)
            out.release()

        # Log the scene information
        obj_list_dest = os.path.join(subdir_path, "object_list.txt")
        task_list_dest = os.path.join(subdir_path, "task_list.txt")
        time_dest = os.path.join(subdir_path, "time.txt")
        robot_id_dest = os.path.join(subdir_path, "robot_id.txt")
        with open(obj_list_dest, "w") as f:
            json.dump(self.object_list, f, indent=4)
        with open(task_list_dest, "w") as f:
            json.dump(self.task_list, f, indent=4)
        time = subprocess.check_output("date", shell=True).decode("utf-8").strip()
        with open(time_dest, "w") as f:
            f.write(time)
        robot_id = self.robot_id
        with open(robot_id_dest, "w") as f:
            f.write(robot_id)

        # Reset variables
        self.obs_images = []
        self.goal_images = []
        self.actions = []
        self.poses = []
        self.traj_idx += 1
