from collections.abc import Mapping

import imageio
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import wandb
from flax.core import frozen_dict
from flax.training import checkpoints

from jaxrl_m.vision.bigvision_resnetv2 import load


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


def bigvision_resnet_loader(params, modules, checkpoint_path):
    params = frozen_dict.unfreeze(params)
    for module in modules:
        params[module]["encoder"]["encoder"] = load(
            params[module]["encoder"]["encoder"],
            checkpoint_path,
            dont_load=["head/bias", "head/kernel"],
        )
    return frozen_dict.freeze(params)


def bigvision_resnet_gc_loader(
    params, modules, checkpoint_path, clone_new_weights=True
):
    params = frozen_dict.unfreeze(params)
    for module in modules:
        encoder_params = load(
            params[module]["encoder"]["encoder"],
            checkpoint_path,
            dont_load=["head/bias", "head/kernel"],
        )
        if clone_new_weights:
            encoder_params["root_block"]["conv_root"]["kernel"] = (
                jnp.concatenate(
                    [encoder_params["root_block"]["conv_root"]["kernel"]] * 2, axis=2
                )
                / 2
            )
        else:
            encoder_params["root_block"]["conv_root"]["kernel"] = jnp.concatenate(
                [
                    encoder_params["root_block"]["conv_root"]["kernel"],
                    jnp.zeros_like(encoder_params["root_block"]["conv_root"]["kernel"]),
                ],
                axis=2,
            )

    return frozen_dict.freeze(params)


def calql_dr3_resnet_gc_loader(params, checkpoint_path):
    params = frozen_dict.unfreeze(params)
    restored_agent = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    params["actor"]["encoder"]["encoder"] = restored_agent["actor"]["encoder"][
        "encoder"
    ]
    params["critic"]["encoder"]["encoder"] = restored_agent["critic"]["encoder"][
        "encoder"
    ]
    params = frozen_dict.freeze(params)
    return params


pretrained_loaders = dict(
    bigvision_resnet=bigvision_resnet_loader,
    bigvision_resnet_gc=bigvision_resnet_gc_loader,
    calql_dr3_resnet_gc=calql_dr3_resnet_gc_loader,
)
