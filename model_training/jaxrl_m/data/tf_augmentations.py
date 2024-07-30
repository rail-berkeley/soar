from collections.abc import Mapping

import tensorflow as tf
from ml_collections import ConfigDict


def random_resized_crop(image, scale, ratio, seed, batched=False):
    if not batched:
        image = tf.expand_dims(image, axis=0)
    batch_size = tf.shape(image)[0]
    # taken from https://keras.io/examples/vision/nnclr/#random-resized-crops
    log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
    height = tf.shape(image)[-3]
    width = tf.shape(image)[-2]

    random_scales = tf.random.stateless_uniform((batch_size,), seed, scale[0], scale[1])
    random_ratios = tf.exp(
        tf.random.stateless_uniform((batch_size,), seed, log_ratio[0], log_ratio[1])
    )

    new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
    new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
    height_offsets = tf.random.stateless_uniform(
        (batch_size,), seed, 0, 1 - new_heights
    )
    width_offsets = tf.random.stateless_uniform((batch_size,), seed, 0, 1 - new_widths)

    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    if len(tf.shape(image)) == 5:
        obs_horizon = tf.shape(image)[1]
        # fold obs_horizon dimension into batch dimension
        image = tf.reshape(image, [batch_size * obs_horizon, height, width, -1])
        # repeat bounding_boxes so each obs history is augmented the same
        bounding_boxes = tf.repeat(bounding_boxes, obs_horizon, axis=0)
        image = tf.image.crop_and_resize(
            image, bounding_boxes, tf.range(batch_size * obs_horizon), (height, width)
        )
        image = tf.reshape(image, [batch_size, obs_horizon, height, width, -1])
    else:
        image = tf.image.crop_and_resize(
            image, bounding_boxes, tf.range(batch_size), (height, width)
        )

    if not batched:
        return image[0]
    else:
        return image


AUGMENT_OPS = {
    "random_resized_crop": random_resized_crop,
    "random_brightness": tf.image.stateless_random_brightness,
    "random_contrast": tf.image.stateless_random_contrast,
    "random_saturation": tf.image.stateless_random_saturation,
    "random_hue": tf.image.stateless_random_hue,
    "random_flip_left_right": tf.image.stateless_random_flip_left_right,
}


def augment(image, seed, **augment_kwargs):
    image = tf.cast(image, tf.float32) / 255  # convert images to [0, 1]
    for op in augment_kwargs["augment_order"]:
        if op in augment_kwargs:
            if isinstance(augment_kwargs[op], Mapping) or isinstance(
                augment_kwargs[op], ConfigDict
            ):
                image = AUGMENT_OPS[op](image, seed=seed, **augment_kwargs[op])
            else:
                image = AUGMENT_OPS[op](image, seed=seed, *augment_kwargs[op])
        else:
            image = AUGMENT_OPS[op](image, seed=seed)
        image = tf.clip_by_value(image, 0, 1)
    image = tf.cast(image * 255, tf.uint8)
    return image
