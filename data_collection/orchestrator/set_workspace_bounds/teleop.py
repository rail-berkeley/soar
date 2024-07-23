import yaml
from yamlinclude import YamlIncludeConstructor
from absl import app, flags
import numpy as np
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
import os
import sys
import tty
import termios
import time

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config_dir",
    None,
    "Path to config directory",
    required=True,
)

print_yellow = lambda x: print("\033[93m {}\033[00m" .format(x))

def print_help():
    print_yellow("  Teleop Controls:")
    print_yellow("    w, s : move forward/backward")
    print_yellow("    a, d : move left/right")
    print_yellow("    z, c : move up/down")
    print_yellow("    i, k:  rotate yaw")
    print_yellow("    j, l:  rotate pitch")
    print_yellow("    n  m:  rotate roll")
    print_yellow("    space: toggle gripper")
    print_yellow("    r: reset robot")
    print_yellow("    q: quit")

def main(_):
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=FLAGS.config_dir)
    with open(os.path.join(FLAGS.config_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update({"action_clipping" : None})
    client = WidowXClient(host=config["general_params"]["ip"], port=config["general_params"]["port"])
    client.init(env_params)
    client.reset()

    # Save the terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print_help()
    is_open = 1
    running = True
    xyz_min, xyz_max = None, None
    while running:
        # Check for key press
        try:
            # Set the terminal to raw mode to read a single key press
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            # Restore the terminal to its original settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # escape key to quit
        if key == 'q':
            print("Quitting teleoperation.")
            running = False
            continue

        # Handle key press for robot control
        # translation
        if key == 'w':
            client.step_action(np.array([0.01, 0, 0, 0, 0, 0, is_open]))
        elif key == 's':
            client.step_action(np.array([-0.01, 0, 0, 0, 0, 0, is_open]))
        elif key == 'a':
            client.step_action(np.array([0, 0.01, 0, 0, 0, 0, is_open]))
        elif key == 'd':
            client.step_action(np.array([0, -0.01, 0, 0, 0, 0, is_open]))
        elif key == 'z':
            client.step_action(np.array([0, 0, 0.01, 0, 0, 0, is_open]))
        elif key == 'c':
            client.step_action(np.array([0, 0, -0.01, 0, 0, 0, is_open]))
        
        # rotation
        elif key == 'i':
            client.step_action(np.array([0, 0, 0, 0.01, 0, 0, is_open]))
        elif key == 'k':
            client.step_action(np.array([0, 0, 0, -0.01, 0, 0, is_open]))
        elif key == 'j':
            client.step_action(np.array([0, 0, 0, 0, 0.01, 0, is_open]))
        elif key == 'l':
            client.step_action(np.array([0, 0, 0, 0, -0.01, 0, is_open]))
        elif key == 'n':
            client.step_action(np.array([0, 0, 0, 0, 0, 0.01, is_open]))
        elif key == 'm':
            client.step_action(np.array([0, 0, 0, 0, 0, -0.01, is_open]))    
        
        # space bar to change gripper state
        elif key == ' ':
            is_open = 1 - is_open
            print("Gripper is now: ", is_open)
            client.step_action(np.array([0, 0, 0, 0, 0, 0, is_open]))
        elif key == 'r':
            print("Resetting robot...")
            client.reset()
            print_help()

        # Get the end-effector position after taking action
        obs = client.get_observation()
        eef_pose = obs["state"]
        if xyz_min is None or xyz_max is None:
            xyz_min = eef_pose[:3]
            xyz_max = eef_pose[:3]
        xyz_min = np.minimum(xyz_min, eef_pose[:3])
        xyz_max = np.maximum(xyz_max, eef_pose[:3])
        print("robot pose:", eef_pose)

    client.stop()  # Properly stop the client
    print("Teleoperation ended.")

    print()
    print("XYZ Min:", xyz_min)
    print("XYZ Max:", xyz_max)

if __name__ == "__main__":
    app.run(main)
