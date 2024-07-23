# Robot Configuration

This directory contains a set of configuration files that specify things related to connecting to the robot, logging, image subgoal generation model parameters, goal-conditioned policy architecture, VLM task proposing and success detection, etc.

## config.yaml

This is the top level configuration file that points to other more specific files for all of the aforementioned pipeline components. Make sure to specify configuration files here for the following:

- general_params
- task_proposer_params
- gc_policy_params
- subgoal_predictor_params
- reset_detector_params
- success_detector_params
- cogvlm_server_params
- task_definition_params

## general_params.yaml
This file contains configuration parameters for connecting to the robot, streaming video feed of the robot's third-person camera to a web viewer, workspace boundaries, and logging. 

1. Specify the IP address and port for the machine plugged into the robot. On this machine, you will need to run the bridge data robot server (see installation instructions from https://github.com/rail-berkeley/bridge_data_robot), and the IP address and port should match that of the server. 
2. We also provide for convenience a web UI that allows you to visualize what your robot(s) is/are doing from anywhere. The web UI is a Flask server that you will need to host on your machine. See instructions in the top-level README for how to do this. `web_viewer_ip` and `web_viewer_port` should be set corresponding to your hosted web server. 
3. If you want to run simultaneous autonomous data collection on an arm farm (as we did in the paper) to scale data collection, then each robot will have a unique id which you must specify.
4. Ignore `override_workspace_boundaries` (this is deprecated)
5. `move_duration` is the time alloted for the robot to execute one action. We collect data and evaluate the WidowX robots with blocking control, in which action prediction for the next timestep occurs only after the robot has been given `move_duration` number of seconds to execute the previous action. Longer `move_duration` leads to more precise execution of commanded actions, but slower data collection.
6. `video_save_path` is a local directory where all your collected robot data will be logged. The logged data include image states and robot actions, generated subgoals, task and robot metadata, etc. Data is saved as numpy arrays, which if you want to subsequently use for training you need to convert to RLDS format (see instructions in top-level README for how to do this conversion). 
7. Specifying `manual_workspace_bounds` allows you to define a rectangular prism in which the end-effector is forced to remain. It is a good idea to specify this as it enables much more safe robot behavior during autonomous data collection. The two numbers for `x`, `y`, and `z` correspond to the minimum and maximum allowed values for each dimension. See instructions in the top-level README for a convenience script that allows you to easily determine what these boundary values are.

## task_proposer.yaml

There multiple different types of task proposers implemented in this code base (see orchestrator/robot/task_proposer.py for the implementations). The implemented task proposers include VLM task proposers using CogVLM and GPT4-V/o in which the VLM looks at the current observations and chooses a task to command from a list of possible tasks, a cycling task proposer which for a two-task setup (e.g., opening and closing a drawer) simply cycles between commanding each of the two tasks, and a human task proposer, which periodically queries you to enter a task via the command line. 

The VLM task proposer, in addition to considering environment affordances prescribed by the current image observation, also considers how many times each of the tasks has been attempted. If two tasks are viable to command according to the VLM and one has been attempted more than the other, the less attempted task will be commanded. To enable this, the `reuse_task_statistics` flag controls whether or not to load previous trajectories in the logging folder to compute the number of times each task has been attempted. If set to false, attempt counters for each of the tasks will be initialized to zero. 

`rand_selection_prob` specifies a probability with which to ignore the task proposed by the VLM and instead propose a random task. Setting this to a nonzero number can be useful in cases where the VLM is less accurate.

## gc_bc.yaml

1. Specify `checkpoint_path` to be the directory containing the policy checkpoint you trained or downloaded from huggingface (for the latter see instructions in `checkpoints` folder).
2. `rollout_timesteps` controls how many timesteps to roll out the goal-conditioned policy before a new image subgoal is synthesized.
3. Adding noise diversifies the collected robot data and facilitates exploration. `exploration` controls the parameters of the added noise.
4. `open_gripper_if_nothing_grasped`, if set to true, will force the gripper open (even if the policy is commanding it to be closed) if the encoders on the gripper read that nothing has been grasped. We leave this set to false in all our experiments.
5. `restrict_action_space` restricts the action space to 5 dimensions, assigning the pitch and yaw dimensions to known good values. In particular challenging environments this can help exploration if it is known that these two angle dimensions are not needed to complete desired tasks. This variable is set to false in all of our experiments.
6. `agent_kwargs` controls the configuration of the trained policy. See the source code under the `model_training` for what these configurations control.

# subgoal_predictor.yaml

This config file controls the image generation parameters for SuSIE, the InstructPix2Pix style diffusion model we use for subgoal generation. The defaults should be fine. The only parameters that need to be set are `susie_server_ip` and `susie_server_port`, which should be set appropriately depending on where you host the SuSIE server (see instructions in top-level README for hosting this server). 

# reset_detector.yaml

During autonomous data collection, it is possible that objects may fall out of the workspace boundaries. To prevent the user from having to monitor the robots 24/7, we have included an automatic reset detection functionality, whereby the VLM is used to determine if objects are missing from the workspace, and if so send a slack messsage to a channel you create. Specify `slack_token` and `channel_id` appropriately for the channel you create. `which_vlm` controls which VLM is used for reset detection.

# success_detector.yaml

`which_vlm` controls which VLM you would like to use for determining task success at the end of every trajectory. We used CogVLM for our experiments. 

# cogvlm_server.yaml

Specify here the IP address and port of the machine you are hosting CogVLM on. CogVLM requires around 48 Gb of memory for batched 
inference.

# task_definition.yaml

When setting up your robot in a new evironment, make sure to specify which objects are present in the scene, and what are all of 
the language tasks you want to be run during data collection. In addition to guiding the VLM for task selection, the information here is logged with every collected trajectory. 