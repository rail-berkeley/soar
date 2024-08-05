import os
import random
import time
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint

from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.vision import encoders


class GCPolicy:
    def __init__(self, config):
        self.gc_config = config["gc_policy_params"]
        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": self.gc_config["ACT_MEAN"],
                "std": self.gc_config["ACT_STD"],
                "min": self.gc_config["ACT_MEAN"],  # we don't use this value
                "max": self.gc_config["ACT_STD"],  # we don't use this value
            },
            "proprio": {
                "mean": self.gc_config["ACT_MEAN"],  # we don't use this value
                "std": self.gc_config["ACT_STD"],  # we don't use this value
                "min": self.gc_config["ACT_MEAN"],  # we don't use this value
                "max": self.gc_config["ACT_STD"],  # we don't use this value
            },
        }
        self.action_statistics = {
            "mean": self.gc_config["ACT_MEAN"],
            "std": self.gc_config["ACT_STD"],
        }

        example_batch = {
            "observations": {
                "image": jnp.zeros((1, 256, 256, 3)),
                "proprio": jnp.zeros((1, 7)),
            },
            "goals": {
                "image": jnp.zeros((1, 256, 256, 3)),
                "proprio": jnp.zeros((1, 7)),
            },
            "actions": jnp.zeros((1, 7)),
        }

        encoder_config = self.gc_config["encoder"]
        encoder_def = encoders[encoder_config["type"]](**encoder_config["config"])
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        self.agent = agents[self.gc_config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **self.gc_config["agent_kwargs"],
        )

        self.update_weights()

        self.image_size = self.gc_config["image_size"]

        # Sticky gripper
        self.sticky_gripper_num_steps = config["general_params"][
            "sticky_gripper_num_steps"
        ]
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

        # Workspace bounds
        self.bounds_x = config["general_params"]["manual_workspace_bounds"]["x"]
        self.bounds_y = config["general_params"]["manual_workspace_bounds"]["y"]
        self.bounds_z = config["general_params"]["manual_workspace_bounds"]["z"]
        self.min_xyz = np.array([self.bounds_x[0], self.bounds_y[0], self.bounds_z[0]])
        self.max_xyz = np.array([self.bounds_x[1], self.bounds_y[1], self.bounds_z[1]])

        # Exploration noise
        self.exploration = self.gc_config["exploration"]

    def update_weights(self):
        while True:
            try:
                print("Updating low-level policy checkpoint...")
                resume_path = self.gc_config["checkpoint_path"]
                restored = orbax.checkpoint.PyTreeCheckpointer().restore(
                    resume_path, item=self.agent
                )
                if self.agent is restored:
                    raise FileNotFoundError(
                        f"Cannot load checkpoint from {resume_path}"
                    )
                print("Checkpoint successfully loaded")
                self.agent = restored
                break
            except:
                print("Error loading checkpoint...")
                raise

    def reset(self):
        """
        Reset is called when the task changes.
        """
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

    def __call__(
        self,
        obs_image: np.ndarray,
        goal_image: np.ndarray,
        pose: np.ndarray,
        deterministic=True,
    ):
        assert obs_image.shape == (
            self.image_size,
            self.image_size,
            3,
        ), "Bad input obs image shape"
        assert goal_image.shape == (
            self.image_size,
            self.image_size,
            3,
        ), "Bad input goal image shape"

        action, action_mode = self.agent.sample_actions(
            {"image" : obs_image[np.newaxis, ...]},
            {"image" : goal_image[np.newaxis, ...]}, 
            argmax=deterministic,
            seed=None if deterministic else jax.random.PRNGKey(int(time.time())),
        )
        action, action_mode = np.array(action.tolist()), np.array(action_mode.tolist())
        action, action_mode = action[0], action_mode[0]

        # Remove exploration in unwanted dimensions
        action[3] = action_mode[3]  # yaw
        action[4] = action_mode[4]  # pitch
        action[-1] = action_mode[-1]  # gripper

        # Scale action
        action[:6] = np.array(self.action_statistics["std"][:6]) * action[
            :6
        ] + np.array(self.action_statistics["mean"][:6])
        action_mode[:6] = np.array(self.action_statistics["std"][:6]) * action_mode[
            :6
        ] + np.array(self.action_statistics["mean"][:6])

        # Sticky gripper logic
        if (action[-1] < 0.0) != (self.gripper_state == "closed"):
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.gripper_state == "closed" else 1.0

        # Add gripper noise
        if not deterministic:
            assert self.sticky_gripper_num_steps == 1
            switch_gripper_action_threshold = (
                self.exploration["gripper_open_prob"]
                if action[-1] == 0.0
                else self.exploration["gripper_close_prob"]
            )
            if random.random() < switch_gripper_action_threshold:
                action[-1] = 0.0 if action[-1] == 1.0 else 1.0

        if self.gc_config["open_gripper_if_nothing_grasped"]:
            # If the gripper is completely closed, that means the grasp was unsuccessful. In that case, let's open the gripper
            if pose[-1] < 0.15:
                action[-1] = 1.0

        if self.gc_config["restrict_action_space"]:
            # Turn off pitch and yaw dimensions of gripper action
            action[4] = -0.1 - pose[4]  # reset dimension to known optimal (zero) value
            action[3] = -pose[3]

        # Clip action to satisfy workspace bounds
        min_action = self.min_xyz - pose[:3]
        max_action = self.max_xyz - pose[:3]
        action[:3] = np.clip(action[:3], min_action, max_action)

        return action

    def get_update_step(self):
        return self.agent.state.step


class LCPolicy:
    def __init__(self, config):
        self.gc_config = config["gc_policy_params"]
        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": self.gc_config["ACT_MEAN"],
                "std": self.gc_config["ACT_STD"],
                "min": self.gc_config["ACT_MEAN"],  # we don't use this value
                "max": self.gc_config["ACT_STD"],  # we don't use this value
            },
            "proprio": {
                "mean": self.gc_config["ACT_MEAN"],  # we don't use this value
                "std": self.gc_config["ACT_STD"],  # we don't use this value
                "min": self.gc_config["ACT_MEAN"],  # we don't use this value
                "max": self.gc_config["ACT_STD"],  # we don't use this value
            },
        }
        self.action_statistics = {
            "mean": self.gc_config["ACT_MEAN"],
            "std": self.gc_config["ACT_STD"],
        }

        # We need to apply a function that encodes text with a language model
        self.text_processor = text_processors[self.gc_config["text_processor"]](
            **self.gc_config["text_processor_kwargs"]
        )

        def process_text(batch):
            batch["goals"]["language"] = self.text_processor.encode(
                [s.decode("utf-8") for s in batch["goals"]["language"]]
            )
            return batch

        example_batch = {
            "observations": {
                "image": jnp.zeros((1, 256, 256, 3)),
                "proprio": jnp.zeros((1, 7)),
            },
            "goals": {
                "language": b"sample string",
                "proprio": jnp.zeros((1, 7)),
            },
            "actions": jnp.zeros((1, 7)),
        }
        example_batch = process_text(example_batch)

        encoder_config = self.gc_config["encoder"]
        encoder_def = encoders[encoder_config["type"]](**encoder_config["config"])
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        self.agent = agents[self.gc_config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **self.gc_config["agent_kwargs"],
        )

        self.update_weights()

        self.image_size = self.gc_config["image_size"]

        # Sticky gripper
        self.sticky_gripper_num_steps = config["general_params"][
            "sticky_gripper_num_steps"
        ]
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

        # Workspace bounds
        self.bounds_x = config["general_params"]["manual_workspace_bounds"]["x"]
        self.bounds_y = config["general_params"]["manual_workspace_bounds"]["y"]
        self.bounds_z = config["general_params"]["manual_workspace_bounds"]["z"]
        self.min_xyz = np.array([self.bounds_x[0], self.bounds_y[0], self.bounds_z[0]])
        self.max_xyz = np.array([self.bounds_x[1], self.bounds_y[1], self.bounds_z[1]])

        # Exploration noise
        self.exploration = self.gc_config["exploration"]

    def update_weights(self):
        while True:
            try:
                print("Updating low-level policy checkpoint...")
                resume_path = self.gc_config["checkpoint_path"]
                restored = orbax.checkpoint.PyTreeCheckpointer().restore(
                    resume_path, item=self.agent
                )
                if self.agent is restored:
                    raise FileNotFoundError(
                        f"Cannot load checkpoint from {resume_path}"
                    )
                print("Checkpoint successfully loaded")
                self.agent = restored
                break
            except:
                print("Error loading checkpoint...")
                raise

    def process_text(self, language_instr):
        return self.text_processor.encode([language_instr])[0]

    def reset(self):
        """
        Reset is called when the task changes.
        """
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

    def __call__(
        self,
        obs_image: np.ndarray,
        language_instr: str,
        pose: np.ndarray,
        deterministic=True,
    ):
        assert obs_image.shape == (
            self.image_size,
            self.image_size,
            3,
        ), "Bad input obs image shape"

        temperature = (
            self.exploration["sampling_temperature"] if not deterministic else 0.0
        )
        language_encoding = self.process_text(language_instr)
        action, action_mode = self.agent.sample_actions(
            {"image": obs_image[np.newaxis, ...]},
            {"language": language_encoding[np.newaxis, ...]},
            temperature=temperature,
            argmax=deterministic,
            seed=None if deterministic else jax.random.PRNGKey(int(time.time())),
        )

        action, action_mode = np.array(action.tolist()), np.array(action_mode.tolist())
        action, action_mode = action[0], action_mode[0]

        # Remove exploration in unwanted dimensions
        action[3] = action_mode[3]  # yaw
        action[4] = action_mode[4]  # pitch
        action[-1] = action_mode[-1]  # gripper

        # Scale action
        action = np.array(self.action_statistics["std"]) * action + np.array(
            self.action_statistics["mean"]
        )
        action_mode = np.array(self.action_statistics["std"]) * action_mode + np.array(
            self.action_statistics["mean"]
        )

        # Sticky gripper logic
        # Note again, different threshold depending on which policy you are using
        if (action[-1] < 0.5) != (self.gripper_state == "closed"):
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.gripper_state == "closed" else 1.0

        # Add gripper noise
        if not deterministic:
            assert self.sticky_gripper_num_steps == 1
            switch_gripper_action_threshold = (
                self.exploration["gripper_open_prob"]
                if action[-1] == 0.0
                else self.exploration["gripper_close_prob"]
            )
            if random.random() < switch_gripper_action_threshold:
                action[-1] = 0.0 if action[-1] == 1.0 else 1.0

        if self.gc_config["open_gripper_if_nothing_grasped"]:
            # If the gripper is completely closed, that means the grasp was unsuccessful. In that case, let's open the gripper
            if pose[-1] < 0.15:
                action[-1] = 1.0

        if self.gc_config["restrict_action_space"]:
            # Turn off pitch and yaw dimensions of gripper action
            action[4] = -0.1 - pose[4]  # reset dimension to known optimal (zero) value
            action[3] = -pose[3]

        # Clip action to satisfy workspace bounds
        min_action = self.min_xyz - pose[:3]
        max_action = self.max_xyz - pose[:3]
        action[:3] = np.clip(action[:3], min_action, max_action)

        return action

    def get_update_step(self):
        return self.agent.state.step


gc_policies = {
    "gc_bc": GCPolicy,
    "lc_bc": LCPolicy,
}
