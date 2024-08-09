from collections.abc import Mapping

import imageio
import numpy as np
import tensorflow as tf
import wandb
from flax.core import frozen_dict


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        if isinstance(batches[0][key], Mapping):
            # to concatenate batch["observations"]["image"], etc.
            concatenated[key] = concatenate_batches([batch[key] for batch in batches])
        else:
            concatenated[key] = np.concatenate(
                [batch[key] for batch in batches], axis=0
            ).astype(np.float32)
    return concatenated


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        if isinstance(batch[key], Mapping):
            # to index into batch["observations"]["image"], etc.
            indexed[key] = index_batch(batch[key], indices)
        else:
            indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    indices = np.random.randint(batch["rewards"].shape[0], size=size)
    return index_batch(batch, indices)


def load_recorded_video(
    video_path: str,
):
    with tf.io.gfile.GFile(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)
