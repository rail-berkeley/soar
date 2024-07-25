import os
import io
import random
import time
import requests

import yaml
from yamlinclude import YamlIncludeConstructor
from absl import app, flags
from PIL import Image
import numpy as np
import tensorflow as tf

from susie.jax_utils import initialize_compilation_cache
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
from jaxrl_m.common.wandb import WandBLogger

from orchestrator.robot.gc_policy import gc_policies
from orchestrator.robot.subgoal_predictor import SubgoalPredictor
from orchestrator.robot.task_proposer import task_proposers
from orchestrator.robot import utils
from orchestrator.robot.logger import Logger
from orchestrator.robot.task_success_predictor import SuccessPredictor
from orchestrator.robot.reset_detector import ResetDetector


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config_dir",
    None,
    "Path to config directory",
    required=True,
)

# experiments
flags.DEFINE_string("exp_name", "", "Experiment name.")
flags.DEFINE_list("tag", list(), "Name of experiment")
flags.DEFINE_string("group", None, "Group of the wandb experiments")
flags.DEFINE_string("project", None, "Project of the wandb experiments")
flags.DEFINE_bool("debug", False, "Debug mode, no wandb logging.")


def rollout_subgoal(
    widowx_client,
    low_level_policy,
    image_goal,
    logger,
    config,
    status_info,
    deterministic=False,
    language_instr=None,
):
    t = 0
    try:
        while t < config["gc_policy_params"]["rollout_timesteps"]:
            obs = utils.get_observation(widowx_client, config)
            image_obs, pose = obs["image"], obs["state"]
            image_obs = np.array(
                Image.fromarray(image_obs).resize(
                    (
                        config["gc_policy_params"]["image_size"],
                        config["gc_policy_params"]["image_size"],
                    )
                )
            ).astype(np.uint8)

            # Save curr obs to disk for visualization
            # imageio.imwrite("obs.png", image_obs) # legacy

            if language_instr is None:
                # Also resize the goal image to pass to the low level policy
                goal_image_resized = np.array(
                    Image.fromarray(image_goal).resize(
                        (
                            config["gc_policy_params"]["image_size"],
                            config["gc_policy_params"]["image_size"],
                        )
                    )
                ).astype(np.uint8)
                action = low_level_policy(
                    image_obs, goal_image_resized, pose, deterministic=deterministic
                )
            else:
                # If language_instr is not None, it means our low level policy is a language conditioned policy
                action = low_level_policy(
                    image_obs, language_instr, pose, deterministic=deterministic
                )

            # Log everything
            logger.log_obs(image_obs)
            logger.log_goal(image_goal)
            logger.log_action(action)
            logger.log_pose(pose)

            # Update web viewer
            status_info["timestep"] = t
            url = (
                "http://"
                + config["general_params"]["web_viewer_ip"]
                + ":"
                + str(config["general_params"]["web_viewer_port"])
                + "/update_status/"
                + str(config["general_params"]["robot_id"])
            )
            _ = requests.post(url, json=status_info)

            # Execute action
            widowx_client.step_action(action, blocking=True)

            print(f"Timestep {t}, pose: {pose}")
            t += 1

    except KeyboardInterrupt:
        return True  # trajectory over
    return False


def get_image_from_web_viewer(config):
    while True:
        try:
            url = (
                "http://"
                + config["general_params"]["web_viewer_ip"]
                + ":"
                + str(config["general_params"]["web_viewer_port"])
                + "/images/"
                + str(config["general_params"]["robot_id"])
                + "/observation"
            )
            response = requests.get(url)
            with Image.open(io.BytesIO(response.content)) as img:
                img = np.array(img)
            break
        except:
            pass  # repeat get request in case of error

    return img


def execute_trajectory(
    widowx_client,
    env_params,
    tasker,
    low_level_policy,
    high_level_policy,
    logger,
    success_predictor,
    reset_detector,
    config,
):
    widowx_client.init(env_params)

    # widowx_client.move_gripper(1.0)  # open gripper
    widowx_client.reset()  # Move to initial position
    # initial_eep = config["general_params"]["initial_eep"]
    # print(f"Moving to position {initial_eep}")
    # print(widowx_client.move(utils.state_to_eep(initial_eep[:3], 0), blocking=True))
    # time.sleep(2.0)

    low_level_policy.reset()

    # We will get the current obs to feed into the task proposer from the web viewer
    # as opposed to from widowX, since the image is higher resolution
    time.sleep(3)  # Wait so web viewer catches up with reality
    initial_obs = get_image_from_web_viewer(config)

    # Check if reset is required
    print("checking if reset is required...")
    object_presence_info = reset_detector.detect(initial_obs)
    off_limits_objects = []
    for obj, obj_present in object_presence_info.items():
        if not obj_present:
            off_limits_objects.append(obj)
    off_limits_objects = None if len(off_limits_objects) == 0 else off_limits_objects
    print("off limits objects:", off_limits_objects)

    # Propose task
    print("proposing task...")
    task = tasker.propose_task(initial_obs, off_limits_objects)

    # check if the task is valid and if not, skip to the next traj
    # the task is not valid if the success_predictor considers it success
    # before any policy is executed
    initial_success = success_predictor.predict_outcome(
        initial_obs, task, log_metrics=False
    )

    retry_counter = 0
    while initial_success:
        print("The task proposed has already succeeded.... Moving to next task")
        retry_counter += 1
        task = tasker.propose_task(initial_obs, off_limits_objects)
        initial_success = success_predictor.predict_outcome(
            initial_obs, task, log_metrics=False
        )

        if retry_counter > 3:
            print(
                "All proposed tasks have already succeeded... Just command the current task"
            )
            break

    done = False
    n = 0
    deterministic = (
        random.random()
        < config["gc_policy_params"]["exploration"]["make_traj_deterministic_prob"]
    )
    while not done and n < config["subgoal_predictor_params"]["max_subgoals"]:
        curr_obs = utils.get_observation(widowx_client, config)["image"]
        # Sample goal
        print(f"Sampling goal {n}...")
        print(f"'{task}'")
        print()
        curr_obs = np.array(
            Image.fromarray(curr_obs).resize(
                (
                    config["subgoal_predictor_params"]["image_size"],
                    config["subgoal_predictor_params"]["image_size"],
                )
            )
        ).astype(np.uint8)
        image_goal = high_level_policy(curr_obs, task)

        # Save goal image
        # imageio.imwrite("goal.png", image_goal) # Legacy, we have a web viewer now

        # Send goal image to web viewer
        img = Image.fromarray(image_goal)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        files = {"file": ("image.jpg", buffer.getvalue(), "image/jpeg")}

        url = (
            "http://"
            + config["general_params"]["web_viewer_ip"]
            + ":"
            + str(config["general_params"]["web_viewer_port"])
            + "/upload/"
            + str(config["general_params"]["robot_id"])
            + "?type=goal"
        )
        _ = requests.post(url, files=files)

        # Create status_info json to send to web viewer
        status_info = {
            "commanded_task": task,
            "subgoal": n,
            "task_success": "in-progress",
        }

        # Rollout
        if config["gc_policy_params"]["policy_class"] == "lc_bc":
            language_instr = task
        else:
            language_instr = None
        done = rollout_subgoal(
            widowx_client,
            low_level_policy,
            image_goal,
            logger,
            config,
            status_info,
            deterministic=deterministic,
            language_instr=language_instr,
        )
        n += 1

    # check whether the moter has failed
    try:
        retry_counter = 0
        max_retry = 10
        motor_is_good = sum(widowx_client.get_motor_status()) == 0
        while not motor_is_good and retry_counter < max_retry:
            print("Some moter has failed")
            logger.reset()
            reboot_joints(widowx_client, config)
            motor_is_good = sum(widowx_client.get_motor_status()) == 0
            retry_counter += 1
            time.sleep(1)
            if motor_is_good:
                # skip to the next traj, don't bother evaluating success
                return
        if not motor_is_good:
            # cannot reboot the joints, exit
            exit(1)
    except TypeError:
        # get_motor_status returns None in the middle of the traj
        pass

    # Now we will assess whether the robot was successful in completing the task
    time.sleep(3)  # Wait so web viewer catches up with reality
    final_obs = get_image_from_web_viewer(config)

    print("predicting task success...")
    success = success_predictor.predict_outcome(final_obs, task)

    # Update the task proposer with this information
    completion_info = {"task_str": task, "success": success}
    tasker.log_task_completion(completion_info)

    # We'll also update the web viewer with success information
    final_status_info = {
        "commanded_task": task,
        "subgoal": config["subgoal_predictor_params"]["max_subgoals"],
        "timestep": 0,
        "task_success": "succeeded" if success else "failed",
    }
    url = (
        "http://"
        + config["general_params"]["web_viewer_ip"]
        + ":"
        + str(config["general_params"]["web_viewer_port"])
        + "/update_status/"
        + str(config["general_params"]["robot_id"])
    )
    _ = requests.post(url, json=final_status_info)

    logger.flush_trajectory(task, success)


def reboot_joints(widowx_client, config):
    print("Rebooting joints...")

    # Go to sleep pose
    obs = utils.get_observation(widowx_client, config)
    curr_pose = obs["state"]
    pre_sleep_pose = np.array(
        [0.14252, -0.00244, 0.098033, -0.00539, -0.7836, 0.0055, 1.0]
    )  # corresponds to a position near the sleep pose
    # pre_sleep_pose = np.array([0.33692948, 0.00805364, 0.02624852, -0.00474271, 0.11188847, -0.00563708, 1.0]) # corresponds to a position almost touching the table, near the reset pose
    pre_sleep_action = pre_sleep_pose - curr_pose
    pre_sleep_action[-1] = 1.0
    widowx_client.step_action(pre_sleep_action, blocking=True)
    widowx_client.sleep()  # additionally go to sleep pose since for some reason prev command doesn't fully execute

    # Reboot joints one by one
    joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
    ]
    for joint in joint_names:
        widowx_client.reboot_motor(joint)
        time.sleep(0.5)

    print("Finished")


def main(_):
    tf.config.set_visible_devices([], "GPU")

    YamlIncludeConstructor.add_to_loader_class(
        loader_class=yaml.FullLoader, base_dir=FLAGS.config_dir
    )
    with open(os.path.join(FLAGS.config_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    initialize_compilation_cache()

    # for evals
    if FLAGS.debug:
        config["task_proposer_params"]["which_vlm"] = "human"
        config["reset_detector_params"]["which_vlm"] = "none"
        config["success_detector_params"]["which_vlm"] = "none"
        config["general_params"]["video_save_path"] = "video_logs"
        config["gc_policy_params"]["exploration"]["make_traj_deterministic_prob"] = 1.0
        config["gc_policy_params"]["rollout_timesteps"] = 25

    low_level_policy = gc_policies[config["gc_policy_params"]["policy_class"]](config)
    high_level_policy = SubgoalPredictor(config)

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(config["general_params"]["env_params"])
    widowx_client = WidowXClient(
        host=config["general_params"]["ip"], port=config["general_params"]["port"]
    )

    logger = Logger(config)
    tasker = task_proposers[config["task_proposer_params"]["which_vlm"]](config)
    success_predictor = SuccessPredictor(config)
    reset_detector = ResetDetector(config)

    # wandb logger
    log_to_wandb = config["task_proposer_params"]["which_vlm"] != "human"
    if log_to_wandb:
        wandb_config = WandBLogger.get_default_config()
        if FLAGS.project is None:
            project_name = "autonomous_widowx"
        else:
            project_name = FLAGS.project

        exp_name = FLAGS.exp_name + f"_robot_{config['general_params']['robot_id']}_"
        exp_name = exp_name + "_".join(config["task_definition_params"]["object_list"])
        wandb_config.update(
            {
                "project": project_name,
                "entity": "rail-iterated-offline-rl",
                "exp_descriptor": exp_name,
                "tag": FLAGS.tag,
                "group": FLAGS.group,
            }
        )
        wandb_logger = WandBLogger(
            wandb_config=wandb_config,
            variant=config,
            debug=FLAGS.debug,
        )

    # resume from previously saved data
    if log_to_wandb:
        i = sum(tasker.get_task_attempted_counter().values())
    else:
        i = 0
    while True:
        try:
            execute_trajectory(
                widowx_client,
                env_params,
                tasker,
                low_level_policy,
                high_level_policy,
                logger,
                success_predictor,
                reset_detector,
                config,
            )
            i += 1

            # log success info to wandb
            if log_to_wandb:
                overall_success_rate = success_predictor.get_success_rate()
                recent_success_rate = success_predictor.get_success_rate(
                    n_most_recent=10
                )
                n_attempted = tasker.get_task_attempted_counter()
                per_task_info = {
                    task: {
                        "overall_success_rate": overall_success_rate[task],
                        "recent_success_rate": recent_success_rate[task],
                        "attempted_counter": n_attempted[task],
                        "policy_update_step": low_level_policy.get_update_step(),
                    }
                    for task in overall_success_rate.keys()
                }
                wandb_logger.log(per_task_info, step=i)

            # periodically reset the robot joints to avoid problems
            print("Periodically reset the robot joints")
            reboot_interval = config["general_params"].get("joints_reboot_interval", 10)
            if i % reboot_interval == 0:
                reboot_joints(widowx_client, config)

            # Load the new checkpoint weights
            # low_level_policy.update_weights()

        except KeyboardInterrupt:
            # manually reboot the robot when you see a motor fail
            logger.reset()
            need_reboot = input("Do you want to reboot the joints? (y/n): ")
            if need_reboot == "y":
                # Reboot joints
                reboot_joints(widowx_client, config)

            continue


if __name__ == "__main__":
    app.run(main)
