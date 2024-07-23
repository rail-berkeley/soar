import random
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags
from tpu_utils import prevent_cross_region

from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import BridgeDataset
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.utils.visualization_utils import value_and_reward_visulization
from jaxrl_m.utils.train_utils import pretrained_loaders
from jaxrl_m.vision import encoders

try:
    from jax_smi import initialise_tracking  # type: ignore
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "", "Experiment name.")
flags.DEFINE_list('tag', list(), 'Name of experiment')
flags.DEFINE_string('group', None, 'Group of the wandb experiments')
flags.DEFINE_bool("debug", False, "Debug config")
flags.DEFINE_integer("utd", 1, "update to data ratio")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)

def main(_):
    # prevent cross region transfer
    prevent_cross_region(
        *FLAGS.bridgedata_config.pretraining_data,
        *FLAGS.bridgedata_config.autonomous_data,
        FLAGS.config.save_dir,
    )

    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            # "project": f"jaxrl_{FLAGS.config.agent}_autonomous_data",
            "project": f"jaxrl_{FLAGS.config.agent}_bridge",
            "exp_descriptor": FLAGS.exp_name,
            "tag": FLAGS.tag,
            "group": FLAGS.group,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=FLAGS.debug,
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )


    # load datasets
    random.seed(FLAGS.config.seed)
    train_paths = []
    if FLAGS.bridgedata_config.sampling_weights.pretraining_data > 0:
        train_paths += [FLAGS.bridgedata_config.pretraining_data]
    if FLAGS.bridgedata_config.sampling_weights.autonomous_data_successes > 0:
        train_paths += [FLAGS.bridgedata_config.autonomous_data]
    if FLAGS.bridgedata_config.sampling_weights.autonomous_data_failures > 0:
        train_paths += [FLAGS.bridgedata_config.autonomous_data]

    # create sample weights for training
    train_sample_weights = [
        FLAGS.bridgedata_config.sampling_weights["pretraining_data"],
        FLAGS.bridgedata_config.sampling_weights["autonomous_data_successes"],
        FLAGS.bridgedata_config.sampling_weights["autonomous_data_failures"],
    ]
    train_sample_weights = [weight for weight in train_sample_weights if weight > 0]  # remove 0s from the sample weights
    assert sum(train_sample_weights) == 1.0, f"Sample weights must sum to 1.0, got {sum(train_sample_weights)}"

    # pick out the splits needed from the dataset
    train_data_splits = []
    if FLAGS.bridgedata_config.sampling_weights.pretraining_data > 0:
        train_data_splits.append('train')
    if FLAGS.bridgedata_config.sampling_weights.autonomous_data_successes > 0:
        train_data_splits.append('success')
    if FLAGS.bridgedata_config.sampling_weights.autonomous_data_failures > 0:
        train_data_splits.append('failure')

    train_data = BridgeDataset(
        train_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        train=True,
        sample_weights=train_sample_weights,
        data_splits=train_data_splits,
        paths_index_with_old_tf_format=list(range(1, len(train_paths))),
        **FLAGS.config.dataset_kwargs,
    )
    val_full_traj_data = BridgeDataset(
        FLAGS.bridgedata_config.pretraining_data,
        FLAGS.config.seed,
        batch_size=1,  # return 1 traj because trajs can have different lens
        return_entire_trajectory=True,  # entire traj for plotting Q values
        sample_weights=None,
        data_splits=['val'],
        train=False,
        **FLAGS.config.dataset_kwargs,
    )
    val_data = BridgeDataset(
        FLAGS.bridgedata_config.pretraining_data,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        train=False,
        sample_weights=None,
        data_splits=['val'],
        **FLAGS.config.dataset_kwargs,
    )

    train_data_iter = map(shard_fn, train_data.iterator())

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )

    # define encoder
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )
    for loader_name, loader_kwargs in FLAGS.config.pretrained_loaders:
        loader = partial(pretrained_loaders[loader_name], **loader_kwargs)
        agent = agent.replace(state=agent.state.replace(params=loader(agent.state.params)))
    if FLAGS.config.get("resume_path", "") != "":
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps - agent.state.step))):
        try:
            timer.tick("total")

            timer.tick("dataset")
            batch = shard_batch(next(train_data_iter), sharding)
            timer.tock("dataset")

            timer.tick("train")
            if FLAGS.utd > 1:
                agent, update_info = agent.update_high_utd(batch, utd_ratio=FLAGS.utd)
            else:
                agent, update_info = agent.update(batch)
            timer.tock("train")

            if agent.state.step % FLAGS.config.eval_interval == 0:
                logging.info("Validation...")
                timer.tick("val")

                # plot debug metrics of validation data
                val_metrics = []
                j = 0
                val_iter = map(shard_fn, val_data.iterator())
                for val_batch in val_iter:
                    rng, val_rng = jax.random.split(rng)
                    val_metrics.append(agent.get_debug_metrics(val_batch, seed=val_rng))
                    j += 1
                    if j >= FLAGS.config.num_val_batches:
                        break
                val_metrics = jax.tree_map(lambda *xs: np.mean(xs), *val_metrics)
                wandb_logger.log({"validation": val_metrics}, step=agent.state.step)

                # collect full validation trajs
                val_data_iter = val_full_traj_data.iterator()  # batch size 1, cannot shard
                val_trajs = []
                for _ in range(FLAGS.config.num_val_trajs):
                    traj = next(val_data_iter)
                    val_trajs.append(traj)
                val_trajs = jax.tree_map(lambda x: x[0, ...], val_trajs)  # get rid of batch dim
                # plot the Q values
                wandb_logger.log(
                    {
                        f"evaluation/visualization": wandb.Image(
                            value_and_reward_visulization(val_trajs, agent)
                        )
                    }, step=agent.state.step
                )
                timer.tock("val")

            if agent.state.step % FLAGS.config.save_interval == 0:
                logging.info("Saving checkpoint...")
                checkpoint_path = checkpoints.save_checkpoint(
                    save_dir, agent, step=agent.state.step, keep=1e6
                )
                logging.info("Saved checkpoint to %s", checkpoint_path)

            timer.tock("total")

            if agent.state.step % FLAGS.config.log_interval == 0:
                update_info = jax.device_get(update_info)
                wandb_logger.log({"training": update_info}, step=agent.state.step)

                wandb_logger.log({"timer": timer.get_average_times()}, step=agent.state.step)
        except tf.errors.OpError as e:
            # sometimes tfds will have trouble communicating with cloud storage bucket for some reason...
            print(f"Error in iteration {i}: {e}")
            print("Skipping to next iteration...")
            traceback.print_exc()

            # avoid timer tock errors
            timer.force_tock_everything()

            continue

        except ValueError as e:
            # sometimes wandb will log NaNs
            print(update_info)
            raise e

if __name__ == "__main__":
    app.run(main)
