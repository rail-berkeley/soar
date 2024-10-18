import fnmatch
import os
from functools import partial
from typing import Iterable, List, Optional, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from jax import tree_map
from octo.data.utils.data_utils import get_dataset_statistics
from octo.utils.spec import ModuleSpec

from jaxrl_m.data.tf_augmentations import augment
from jaxrl_m.data.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS


def glob_to_path_list(
    glob_strs: Union[str, List[str]],
    prefix: str = "",
    exclude: Iterable[str] = (),
    exclude_quietly: bool = False,
):
    """Converts a glob string or list of glob strings to a list of paths."""
    if isinstance(glob_strs, str):
        glob_strs = [glob_strs]
    path_list = []
    for glob_str in glob_strs:
        paths = tf.io.gfile.glob(os.path.join(prefix, glob_str))
        filtered_paths = []
        for path in paths:
            if not any(fnmatch.fnmatch(path, e) for e in exclude):
                filtered_paths.append(path)
            else:
                if not exclude_quietly:
                    logging.info(f"Excluding {path}")
        if len(filtered_paths) == 0:
            print("Warning: glob_to_path_list didn't find any paths")
        path_list += filtered_paths
    return path_list


@tf.function(jit_compile=True)
def _binarize_gripper_actions(actions):
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near
    1.0) or fully closed (near 0.0). As it transitions between the two, it
    sometimes passes through a few intermediate values. We relabel those
    intermediate values based on the state that is reached _after_ those
    intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we
    give up on binarizing and relabel that chunk of intermediate values as
    the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, tf.float32),
            lambda: is_open_float[i],
        )

    new_actions = tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )

    return new_actions


def _binary_gripper_to_continuous(
    g_actions,
    gripper_action_mean=0.5,
    gripper_action_std=0.1,
    action_clip_delta=1e-3,
):
    # transform the actions from [0, 1] binary to [-1, 1] continuous
    # and clip the actions to be within bounds to avoid NaNs
    is_open_mask = g_actions > 0.5
    is_open_mask = tf.cast(is_open_mask, tf.float32)
    traj_len = tf.shape(g_actions)[0]

    normal_samples = tf.random.normal(
        [traj_len], mean=gripper_action_mean, stddev=gripper_action_std
    )
    new_actions = normal_samples * is_open_mask + (
        normal_samples - gripper_action_mean * 2
    ) * (1 - is_open_mask)

    new_actions = tf.clip_by_value(
        new_actions,
        -1.0 + action_clip_delta,
        1.0 - action_clip_delta,
    )

    return new_actions


class WidowXDataset:
    """
    Fast parallel tf.data.Dataset-based dataloader for BridgeData and SOAR-Data.
    This format consists of TFRecords where each example
    is one trajectory. See `PROTO_TYPE_SPEC` below for the expected format
    for each example in more detail. See `_process_trajectory` below for
    the output format.

    Includes goal relabeling, image augmentations, and sampling from multiple
    datasets with different weights. Goal relabeling uses a 0/-1 reward scheme:
    0 when the next_obs is labeled as the goal, -1 otherwise.

    Args:
        data_prefixes: List of data prefix to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        seed: Random seed.
        skip_normalization: Whether to skip normalization of actions and proprio.
        normalization_type: The type of normalization to apply to the actions
            and proprio.
        action_clip_delta: If normalization bounds the agent to certain range, this
            clips the action to be within the bounds for another small delta value.
        relabel_actions: Whether to relabel the actions with reached states
            (based on proprioception). Also binarizes gripper actions.
        goal_relabeling_strategy: Goal relabeling strategy. See
            `jaxrl_m.data.tf_goal_relabeling` for more details.
        goal_relabeling_kwargs: Keyword arguments for goal relabeling. See
            `jaxrl_m.data.tf_goal_relabeling` for more details.
        sample_weights: If data_paths is a list of list of paths, this is a
            list of weights with which to sample from each sub-list.
        batch_size: Batch size.
        shuffle_buffer_size: Size of the shuffle buffer. It is split between
            sub-datasets by `sample_weights`.
        cache: Whether to cache the dataset in memory.
        train: Whether this dataset is intended for training
            (if set to `False`, will disable shuffling and augmentations).
        augment: Whether to apply image augmentations.
        augment_kwargs: Keyword arguments for image augmentations. See
            `jaxrl_m.data.tf_augmentations.augment` for more details.
        augment_next_obs_goal_differently: Whether to use different random seeds
            for augmenting the obs, next_obs, and goal image.
        act_pred_horizon: Number of consecutive actions that will be predicted.
        obs_horizon: Number of consecutive observations that will be conditioned on.
        load_langauge: Whether to look for and load language from the data.
        skip_unlabeled: Whether to filter out trajectories not labeled with language.
        gripper_action_mean: Mean of the continuous gripper action distribution.
        gripper_action_std: Standard deviation of the continuous gripper action distribution.
        return_entire_trajectory: Whether to return the entire trajectory as a batch.
        action_merge_horizon: if > 1, sum actions over this many steps.
    """

    def __init__(
        self,
        data_prefixes: List[Union[str, List[str]]],
        seed: int,
        skip_normalization: Optional[bool] = False,
        normalization_type: Optional[str] = "normal",
        action_clip_delta: float = 0,
        relabel_actions: bool = True,
        goal_relabeling_strategy: Optional[str] = None,
        goal_relabeling_kwargs: dict = {},
        sample_weights: Optional[List[float]] = None,
        data_splits: Optional[List[str]] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 10000,
        cache: bool = False,
        train: bool = True,
        augment: bool = False,
        augment_next_obs_goal_differently: bool = False,
        augment_kwargs: dict = {},
        act_pred_horizon: Optional[int] = None,
        obs_horizon: Optional[int] = None,
        load_language: bool = False,
        skip_unlabeled: bool = False,
        gripper_action_mean: float = 0.5,
        gripper_action_std: float = 0.1,
        return_entire_trajectory: bool = False,
        action_merge_horizon: int = 1,
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to WidowXDataset: %s", kwargs)
        if isinstance(data_prefixes[0], str):
            data_prefixes = [data_prefixes]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_prefixes)] * len(data_prefixes)
        assert len(data_prefixes) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)
        if data_splits is None:
            data_splits = [None] * len(data_prefixes)
        else:
            assert len(data_splits) == len(data_prefixes)
            assert all(
                split in ["train", "val", "success", "failure"] for split in data_splits
            ), f"Invalid data split: {data_splits}"

        self.data_splits = data_splits
        self.relabel_actions = relabel_actions
        self.skip_normalization = skip_normalization
        self.normalization_type = normalization_type
        self.action_clip_delta = action_clip_delta
        self.goal_relabeling_strategy = goal_relabeling_strategy
        self.goal_relabeling_kwargs = goal_relabeling_kwargs
        self.cache = cache
        self.augment_kwargs = augment_kwargs
        self.augment_next_obs_goal_differently = augment_next_obs_goal_differently
        self.act_pred_horizon = act_pred_horizon
        self.obs_horizon = obs_horizon
        self.is_train = train
        self.load_language = load_language
        self.gripper_action_mean = gripper_action_mean
        self.gripper_action_std = gripper_action_std
        self.return_entire_trajectory = return_entire_trajectory
        self.action_merge_horizon = action_merge_horizon

        if self.load_language:
            self.PROTO_TYPE_SPEC[
                "steps/language_instruction"
            ] = tf.io.FixedLenSequenceFeature(
                [],
                dtype=tf.string,
                default_value="default",
                allow_missing=True,
            )

        # construct a dataset for each sub-list of paths
        datasets = []

        for i, sub_data_prefix in enumerate(data_prefixes):
            datasets.append(
                self._construct_tf_dataset(
                    sub_data_prefix,
                    data_splits[i],
                    seed,
                )
            )

        if train:
            # shuffle and repeat each sub-dataset, allocating the shuffle buffer
            # by sample_weights
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
                    .repeat()
                )

        # for validation, we want to be able to iterate through the entire dataset;
        # for training, we want to make sure that no sub-dataset is ever exhausted
        # or the sampling ratios will be off. this should never happen because of the
        # repeat() above, but `stop_on_empty_dataset` is a safeguard
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=train
        )

        if skip_unlabeled:
            dataset = dataset.filter(
                lambda x: tf.math.reduce_any(x["goals"]["language"] != "")
            )

        if train and augment:
            # apply augmentations, using a sequence of integers as seeds.
            # this was the only way I found to avoid a memory leak in tf.random.Generator
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=not train,
        )

        self.tf_dataset = dataset

    def _construct_tf_dataset(
        self,
        prefixes: List[str],
        data_split: str,
        seed: int,
    ) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """
        # get the data split paths
        paths = []
        if data_split is None:
            data_split = ""
        for prefix in prefixes:
            paths += glob_to_path_list(
                [f"*{data_split}*.tfrecord*"],
                prefix,
            )

        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(
            self._relabel_actions, num_parallel_calls=tf.data.AUTOTUNE
        )

        # calculate the dataset statistics on the fly (after action relabeling)
        if not self.skip_normalization:
            action_proprio_metadata = self._compute_dataset_statistics(prefixes)

        # yields trajectories
        dataset = dataset.map(
            partial(
                self._process_actions,
                normalization_metadata=action_proprio_metadata,
                merge_actions=self.action_merge_horizon > 1,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # yields trajectories
        dataset = dataset.map(self._chunk_act_obs, num_parallel_calls=tf.data.AUTOTUNE)

        # cache before add_goals because add_goals introduces randomness
        if self.cache:
            dataset = dataset.cache()

        # yields trajectories
        dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to yield individual transitions
        if not self.return_entire_trajectory:
            dataset = dataset.unbatch()

        return dataset

    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "steps/observation/image_0": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.string, allow_missing=True
        ),  # Encoded images as a sequence
        "steps/observation/state": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),  # States as a sequence
        "steps/action": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),  # Actions as a sequence
    }

    def _decode_example(self, serialized_example):
        # Parse the serialized example
        parsed_features = tf.io.parse_single_example(
            serialized_example, self.PROTO_TYPE_SPEC
        )

        # Decode images
        images_decoded = tf.map_fn(
            fn=lambda x: tf.io.decode_jpeg(x, channels=3),
            elems=parsed_features["steps/observation/image_0"],
            fn_output_signature=tf.TensorSpec(shape=[256, 256, 3], dtype=tf.uint8),
        )

        # No need to reshape images since we're already setting the shape in map_fn,
        # but we will have to reshape the states and actions
        states_reshaped = tf.reshape(
            parsed_features["steps/observation/state"], [-1, 7]
        )
        actions_reshaped = tf.reshape(parsed_features["steps/action"], [-1, 7])

        # language instruction
        language_instruction = tf.strings.as_string(
            parsed_features["steps/language_instruction"]
        )

        # restructure the dictionary into the downstream format
        return {
            "observations": {
                "image": images_decoded[:-1],
                "proprio": states_reshaped[:-1],
            },
            "next_observations": {
                "image": images_decoded[1:],
                "proprio": states_reshaped[1:],
            },
            "actions": actions_reshaped[:-1],
            "terminals": tf.zeros_like(actions_reshaped[:-1][:, 0]),
            **({"language": language_instruction[:-1]} if self.load_language else {}),
        }

    def _process_actions(
        self,
        traj,
        normalization_metadata,
        merge_actions=False,
    ):
        # sum actions over a horizon
        if merge_actions:
            assert self.action_merge_horizon > 1
            print(f"NOTE: Summing actions over {self.action_merge_horizon} steps")
            traj = self._sum_actions(traj)
        else:
            assert self.action_merge_horizon == 1

        # normalize actions and proprio
        traj = self._normalize_actions_proprio(traj, normalization_metadata)

        # distribute the discrete gripper action to a normal distribution
        new_gripper_actions = _binary_gripper_to_continuous(
            traj["actions"][:, 6],
            gripper_action_mean=self.gripper_action_mean,
            gripper_action_std=self.gripper_action_std,
            action_clip_delta=self.action_clip_delta,
        )
        traj["actions"] = tf.concat(
            [traj["actions"][:, :6], new_gripper_actions[:, None]], axis=1
        )

        return traj

    def _relabel_actions(self, traj):
        if self.relabel_actions:
            if self.action_merge_horizon == 1:
                # relabel the first 6 action dims (xyz position, xyz rotation)
                # using the reached proprio
                movement_actions = (
                    traj["next_observations"]["proprio"][:, :6]
                    - traj["observations"]["proprio"][:, :6]
                )
            else:
                assert self.action_merge_horizon > 1
                # already extracted movement actions from proprio in _sum_actions()
                movement_actions = traj["actions"][:, :6]

            # binarize the gripper action
            gripper_actions = traj["actions"][:, 6]
            binarized_gripper_actions = _binarize_gripper_actions(
                gripper_actions,
            )

            traj["actions"] = tf.concat(
                [movement_actions, binarized_gripper_actions[:, None]],
                axis=1,
            )

        return traj

    def _compute_dataset_statistics(self, dataset_dir):
        """
        load or compute dataset statistics on the fly
        imported from the octo loaders
        """

        builder = tfds.builder_from_directories(dataset_dir)
        force_recompute_dataset_statistics = True
        filter_functions = ()
        ignore_error = False
        proprio_obs_key = "proprio"
        standardize_fn = ModuleSpec.create(
            "octo.data.oxe.oxe_standardization_transforms:bridge_dataset_transform"
        )

        def is_nonzero_length(traj):
            return tf.shape(traj["action"])[0] > 0

        def standardize(traj):
            # createdy by zhouzypaul
            if standardize_fn is not None:
                traj = ModuleSpec.instantiate(standardize_fn)(traj)

            return traj

        full_dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False)
        for filter_fcn_spec in filter_functions:
            full_dataset = full_dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
        if ignore_error:
            full_dataset = full_dataset.ignore_errors()
        full_dataset = full_dataset.traj_map(standardize).filter(is_nonzero_length)

        # tries to load from cache, otherwise computes on the fly

        # octo code has a bug and does not check for creation of log_dir
        # hence we need to do it here
        default_cache_dir = os.path.expanduser("~/.cache/octo")
        if not os.path.exists(default_cache_dir):
            os.makedirs(default_cache_dir)

        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(proprio_obs_key),
                ModuleSpec.to_string(standardize_fn)
                if standardize_fn is not None
                else "",
                *map(ModuleSpec.to_string, filter_functions),
            ),
            # save_dir=builder.data_dir,
            force_recompute=force_recompute_dataset_statistics,
        )

        dataset_statistics = tree_map(
            np.array, dataset_statistics, is_leaf=lambda x: isinstance(x, list)
        )
        return dataset_statistics

    def _normalize_actions_proprio(self, traj, action_proprio_metadata):
        # normalize actions and proprio
        if action_proprio_metadata is not None:
            if (
                self.normalization_type == "normal"
                or self.normalization_type == "tanh_normal"
            ):
                # normalize to mean 0, std 1
                traj["actions"] = tf.concat(
                    [
                        (
                            traj["actions"][:, :6]
                            - action_proprio_metadata["action"]["mean"][:6]
                        )
                        / action_proprio_metadata["action"]["std"][:6],
                        traj["actions"][:, 6:],
                    ],
                    axis=1,
                )
                if self.normalization_type == "tanh_normal":
                    traj["actions"] = tf.concat(
                        [traj["actions"][:, :6] / 4, traj["actions"][:, 6:]], axis=1
                    )  # makes prob of <-1 or >1 practically 0, as if we tanh'ed it
                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"]
                        - action_proprio_metadata["proprio"]["mean"]
                    ) / action_proprio_metadata["proprio"]["std"]
            elif self.normalization_type in ("bounds", "tanh"):
                # normalize to [0, 1]
                traj["actions"] = tf.concat(
                    [
                        (
                            traj["actions"][:, :6]
                            - action_proprio_metadata["action"]["min"][:6]
                        )
                        / (
                            action_proprio_metadata["action"]["max"][:6]
                            - action_proprio_metadata["action"]["min"][:6]
                        ),
                        traj["actions"][:, 6:],
                    ],
                    axis=1,
                )

                if self.normalization_type == "tanh":
                    # normalize to [-1, 1]
                    traj["actions"] = tf.concat(
                        [
                            traj["actions"][:, :6] * 2 - 1,
                            traj["actions"][:, 6:],
                        ],
                        axis=1,
                    )
                    traj["actions"] = tf.clip_by_value(
                        traj["actions"],
                        -1 + self.action_clip_delta,
                        1 - self.action_clip_delta,
                    )
                elif self.normalization_type == "bounds":
                    # clip to [0, 1]
                    traj["actions"] = tf.clip_by_value(
                        traj["actions"],
                        0 + self.action_clip_delta,
                        1 - self.action_clip_delta,
                    )

                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"] - action_proprio_metadata["proprio"]["min"]
                    ) / (
                        action_proprio_metadata["proprio"]["max"]
                        - action_proprio_metadata["proprio"]["min"]
                    )
                    if self.normalization_type == "tanh":
                        # normalize to [-1, 1]
                        traj[key]["proprio"] = traj[key]["proprio"] * 2 - 1
                        traj[key]["proprio"] = tf.clip_by_value(
                            traj[key]["proprio"],
                            -1 + self.action_clip_delta,
                            1 - self.action_clip_delta,
                        )
                    elif self.normalization_type == "bounds":
                        traj[key]["proprio"] = tf.clip_by_value(
                            traj[key]["proprio"],
                            0 + self.action_clip_delta,
                            1 - self.action_clip_delta,
                        )
            else:
                raise ValueError

        return traj

    def _sum_actions(self, traj):
        """
        sum the adjacent self.action_merge_horizon actions together
        this function is used when self.action_merge_horizon > 1
        """
        if self.relabel_actions:
            # actions are based on proprio
            movement_actions = (
                traj["next_observations"]["proprio"][
                    self.action_merge_horizon - 1 :, :6
                ]
                - traj["observations"]["proprio"][
                    : -(self.action_merge_horizon - 1), :6
                ]
            )
            gripper_actions = traj["actions"][self.action_merge_horizon - 1 :, 6]
            traj["actions"] = tf.concat(
                [movement_actions, gripper_actions[:, None]], axis=1
            )

        else:
            # actions are based on recorded actions

            # Workaround using tf.vectorized_map to sum actions and apply conditional logic
            def sum_and_apply_last_gripper(action_slice):
                summed_action = tf.reduce_sum(action_slice, axis=0)
                gripper_val = action_slice[
                    -1, -1
                ]  # the last gripper value in the slice
                return tf.concat([summed_action[:6], [gripper_val]], axis=0)

            # Sum actions over window and apply conditional logic for gripper
            unstacked_summed_actions = tf.vectorized_map(
                sum_and_apply_last_gripper,
                tf.signal.frame(
                    traj["actions"],
                    frame_length=self.action_merge_horizon,
                    frame_step=1,
                    pad_end=False,
                    axis=0,
                ),
            )
            traj["actions"] = unstacked_summed_actions

        # shorten the horizon for everything
        for k in traj["observations"]:
            traj["observations"][k] = traj["observations"][k][
                : -(self.action_merge_horizon - 1)
            ]
        for k in traj["next_observations"]:
            traj["next_observations"][k] = traj["next_observations"][k][
                self.action_merge_horizon - 1 :
            ]
        traj["terminals"] = traj["terminals"][self.action_merge_horizon - 1 :]

        return traj

    def _chunk_act_obs(self, traj):
        traj_len = len(traj["actions"])
        if self.act_pred_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(self.act_pred_horizon), [traj_len, self.act_pred_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.act_pred_horizon]
            )
            # pads by repeating the last action
            chunk_indices = tf.minimum(chunk_indices, traj_len - 1)
            traj["action_chunks"] = tf.gather(traj["actions"], chunk_indices)
        if self.obs_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(-self.obs_horizon + 1, 1), [traj_len, self.obs_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.obs_horizon]
            )
            # pads by repeating the first observation
            chunk_indices = tf.maximum(chunk_indices, 0)
            traj["obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["observations"]
            )
            traj["next_obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["next_observations"]
            )
        return traj

    def _add_goals(self, traj):
        if self.goal_relabeling_strategy is not None:
            traj = GOAL_RELABELING_FUNCTIONS[self.goal_relabeling_strategy](
                traj, **self.goal_relabeling_kwargs
            )

        if self.load_language:
            lang_idx = tf.random.uniform(
                shape=[], maxval=len(traj["language"]), dtype=tf.int32
            )
            lang = traj["language"][lang_idx]
            traj["goals"]["language"] = tf.broadcast_to(
                lang, tf.shape(traj["terminals"])
            )
            traj.pop("language")

            # always make the "goal" the last obs so that masking is done
            # properly below
            traj_len = tf.shape(traj["goal_dists"])[0]
            traj["goal_dists"] = traj_len - tf.range(traj_len)

        # after goal relabeling, we can set actions and obs to chunked version
        if "action_chunks" in traj:
            traj["actions"] = traj.pop("action_chunks")
            # set movement actions to 0 after the goal is reached
            new_movement = tf.where(
                (
                    traj["goal_dists"][:, None, None]
                    > tf.range(self.act_pred_horizon)[None, :, None]
                ),  # shape (traj_len, act_pred_horizon, 1)
                traj["actions"][
                    :, :, :-1
                ],  # shape (traj_len, act_pred_horizon, action_dim - 1)
                tf.zeros_like(traj["actions"][0, 0, :-1]),  # shape (action_dim - 1)
            )
            # for gripper actions, repeat the last action after the goal is reached
            new_gripper = tf.where(
                (
                    traj["goal_dists"][:, None]
                    > tf.range(self.act_pred_horizon)[None, :]
                ),  # shape (traj_len, act_pred_horizon)
                traj["actions"][:, :, -1],  # shape (traj_len, act_pred_horizon)
                tf.gather(
                    # shifts `actions` to the right by one, padding with the first action
                    tf.concat(
                        [
                            tf.concat(
                                [
                                    traj["actions"][:1, :1, -1],
                                    traj["actions"][:1, :-1, -1],
                                ],
                                axis=1,
                            ),
                            traj["actions"][:-1, :, -1],
                        ],
                        axis=0,
                    ),
                    # selects the action at index `goal_dists` in the previous action chunk
                    tf.minimum(traj["goal_dists"], self.act_pred_horizon - 1),
                    batch_dims=1,
                )[:, None],
            )
            traj["actions"] = tf.concat([new_movement, new_gripper[:, :, None]], axis=2)
        if "obs_chunks" in traj:
            traj["observations"] = traj.pop("obs_chunks")
            traj["next_observations"] = traj.pop("next_obs_chunks")

        return traj

    def _augment(self, seed, image):
        if self.augment_next_obs_goal_differently:
            sub_seeds = tf.unstack(
                tf.random.stateless_uniform(
                    [3, 2], seed=[seed, seed], minval=None, maxval=None, dtype=tf.int32
                )
            )
        else:
            # use the same seed for obs, next_obs, and goal
            sub_seeds = [[seed, seed]] * 3

        for key, sub_seed in zip(
            ["observations", "next_observations", "goals"], sub_seeds
        ):
            image[key]["image"] = augment(
                image[key]["image"], sub_seed, **self.augment_kwargs
            )
        return image

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
