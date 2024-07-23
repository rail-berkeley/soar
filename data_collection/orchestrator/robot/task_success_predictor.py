import os

from tqdm import tqdm
import numpy as np

import utils

class SuccessPredictor:
    """
    Uses a VLM to predict whether a given task has been completed
    """
    def __init__(self, config):
        self.config = config

        # Prompt to convert task description into VQA style question that can be used to assess task completion
        self.task_to_vqa_prompt = """I have a robot arm manipulator in a lab setting that can perform many manipulation tasks. I commanded a task to the robot to perform and I now want to assess whether the robot was successful in completing this task.\n\nTo determine whether or not the robot successfully completed the task, I have access to a vision-language model (VLM) that can answer questions for me when I provide it an image of the lab environment the robot is operating in. Since I can only ask it simple questions about the image, I need to convert the task description into a question that I can feed into the VLM.\n\nHere are several examples:\n\nTask: put the yellow ball on the blue plate\nVLM question: Is the yellow ball on the blue plate?\n\nTask: move the yellow ball from the blue plate to the table\nVLM question: Is the yellow ball on the blue plate or the table?\n\nTask: move the yellow ball to the left side of the table\nVLM question: Is the yellow ball on the left side of the table?\n\nTask: move the yellow ball to the right side of the table\nVLM question: Is the yellow ball on the right side of the table?\n\nTask: put the orange crayon on the blue plate\nVLM question: Is the orange crayon on the blue plate?\n\nTask: move the orange crayon from the blue plate to the table\nVLM question: Is the orange crayon on the blue plate or the table?\n\nTask: put the orange crayon on the cloth\nVLM question: Is the orange crayon on top of the cloth?\n\nTask: move the orange crayon from the cloth to the table\nVLM question: Is the orange crayon on the cloth or the table?\n\nTask: move the orange crayon from the cloth to the blue plate\nVLM question: Is the orange crayon on the cloth or the blue plate?\n\nTask: move the orange crayon from the blue plate to the cloth\nVLM question: Is the orange crayon on the blue plate or the cloth?\n\nTask: move the orange crayon to the right side of the table\nVLM question: Is the orange crayon on the right side of the table?\n\nTask: move the orange crayon to the left side of the table\nVLM question: Is the orange crayon on the left side of the table?\n\nTask: move the red object from the blue plate to the table\nVLM question: Is the red object on the blue plate or the table?\n\nTask: put the red object on the blue plate\nVLM question: Is the red object on the blue plate?\n\nTask: move the red object to the left side of the table\nVLM question: Is the red object on the left side of the table?\n\nTask: move the red object to the right side of the table\nVLM question: Is the red object on the right side of the table?\n\nTask: put the red object on the cloth\nVLM question: Is the red object on the cloth?\n\nTask: move the red object from the cloth to the table\nVLM question: Is the red object on the cloth or the table?\n\nTask: move the red object from the cloth to the blue plate\nVLM question: Is the red object on the cloth or the blue plate?\n\nTask: move the red object from the blue plate to the cloth\nVLM question: Is the red object on the blue plate or the cloth?\n\nTask: move the cloth to the right side of the table\nVLM question: Is the cloth on the right side of the table?\n\nTask: move the cloth to the left side of the table\nVLM question: Is the cloth on the left side of the table?\n\nFollowing the format of these examples, give me the VLM question for the following task:\n\nTask: """

        # Prompt to decode VLM output into true/false
        self.prompt_to_parse_vlm_output = [
            "I have a robot arm manipulator in a lab setting. I commanded it to complete the following task:\n\n",
            "\n\nI want to assess whether the robot arm successfully completed the task. To do so, I prompted a vision-language model (VLM) with an image of the current robot workspace and the following question:\n\n",
            "\n\nIn response, the VLM answered the following:\n\n",
            "\n\nBased on the task commanded and the VLM's response to the question, determine if the robot successfully completed the commanded task or not. If it did successfully complete the task, return just the word true. Otherwise return the word false. If for some reason the answer is neither true nor false, return false."
        ]

        # Use for caching anything you want
        self.cache = {}

        # to record the success rates
        self.task_success_record = self.init_previous_task_stats()  # task -> list of bools

    def init_previous_task_stats(self):
        task_success_record = {}

        
        if self.config["task_proposer_params"]["reuse_task_statistics"]:
            trajectory_log_dir = self.config["general_params"]["video_save_path"]
            logged_trajs = [
                traj for traj in os.listdir(trajectory_log_dir) 
                if os.path.isdir(os.path.join(trajectory_log_dir, traj))
            ]
            for traj in tqdm(logged_trajs):
                traj_path = os.path.join(trajectory_log_dir, traj)
                with open(os.path.join(traj_path, "language_task.txt")) as f:
                    traj_task = f.read().strip().lower()
                with open(os.path.join(traj_path, "success.txt")) as f:
                    traj_success = f.read().strip().lower()
                if traj_task not in task_success_record:
                    task_success_record[traj_task] = []
                task_success_record[traj_task].append(traj_success == "true")
        
        return task_success_record
    
    def record_task_success(self, task_str, success):
        if task_str not in self.task_success_record:
            self.task_success_record[task_str] = []
        self.task_success_record[task_str].append(success)
    
    def get_success_rate(self, n_most_recent=None):
        success_rates = {}
        for task_str, success_list in self.task_success_record.items():
            if n_most_recent is not None:
                success_list = success_list[-n_most_recent:]
            success_rates[task_str] = sum(success_list) / len(success_list)
        return success_rates

    def predict_outcome(self, image: np.ndarray, task_str: str, log_metrics=True):
        # convert the task_str into a VQA style question
        vqa_style_q_unparsed = utils.ask_gpt4(self.task_to_vqa_prompt + task_str, cache=self.cache)

        # add response to cache
        self.cache[self.task_to_vqa_prompt + task_str] = vqa_style_q_unparsed

        vqa_style_q_unparsed = vqa_style_q_unparsed.strip()
        if ":" in vqa_style_q_unparsed:
            vqa_style_q = vqa_style_q_unparsed[vqa_style_q_unparsed.index(":")+2:]
        else:
            vqa_style_q = vqa_style_q_unparsed
        print("vqa_style_q:", vqa_style_q)

        # ask the VLM
        if self.config["success_detector_params"]["which_vlm"] == "gpt4v":
            vlm_output = utils.ask_gpt4v(image, vqa_style_q)
        elif self.config["success_detector_params"]["which_vlm"] == "cogvlm":
            vlm_output = utils.ask_cogvlm(image, vqa_style_q, self.config)
        else:
            # If there's no VLM success detector, we will conservatively assume the task failed
            return False
        print("vlm_output:", vlm_output)

        # parse the output
        decoding_prompt = self.prompt_to_parse_vlm_output[0] + task_str + self.prompt_to_parse_vlm_output[1] + vqa_style_q + self.prompt_to_parse_vlm_output[2] + vlm_output + self.prompt_to_parse_vlm_output[3]
        print("decoding_prompt:", decoding_prompt)
        parsed_vlm_output = utils.ask_gpt4(decoding_prompt, cache=self.cache)

        # add response to cache
        self.cache[decoding_prompt] = parsed_vlm_output

        print("parsed_vlm_output:", parsed_vlm_output)

        success = parsed_vlm_output.strip().lower() == "true"
        try:
            assert success in (True, False)
        except AssertionError:
            print("Error: VLM output was neither 'true' nor 'false'. Assuming task failed.")
            success = False
        
        if log_metrics:
            self.record_task_success(task_str, success)

        return success
