import os
import random
import time
import utils
import math
from abc import ABC, abstractmethod

from tqdm import tqdm
import numpy as np


class AbstractTaskProposer(ABC):
    """
    The most general form of a task proposer might consume 
    many things like the history of previous proposed tasks, 
    prior task success rates, the current image observation, etc.
    To support this, this abstract class defines a task proposer 
    template for subclasses to follow.
    """
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def propose_task(self, image_obs: np.ndarray):
        """Propose a task, possibly looking at the current image observation"""
        pass

    @abstractmethod
    def log_task_completion(self, completion_info: dict):
        """
        Register the completed execution of a task, possibly with info like 
        whether or not the task succeeded
        """
        pass


class ProposerFromTaskList(AbstractTaskProposer):
    """
    Any proposer that requires a pre-defined task list
    """
    def __init__(self, config):
        super().__init__(config)
        self.possible_tasks = self.load_possible_tasks(config)
    
    def load_possible_tasks(self, config):
        self.possible_tasks = []
        if "task_list" in config["task_definition_params"]:
            print("Loading list of tasks from config")
            self.possible_tasks = config["task_definition_params"]["task_list"]
        else:
            print("Enter a list of possible tasks that the robot can perform in this environment (press enter to exit)")
            counter = 1
            while True:
                task = input("(" + str(counter) + "): ")
                if task == "":
                    break
                self.possible_tasks.append(task.strip())
                counter += 1

        # Randomize task ordering
        random.seed(time.time())
        random.shuffle(self.possible_tasks)
        self.possible_task_to_id = {task: i for i, task in enumerate(self.possible_tasks)}
        self.id_to_possible_task = {i: task for i, task in enumerate(self.possible_tasks)}
        return self.possible_tasks
        

class ProposerWithCounters(ProposerFromTaskList):
    """
    Any proposer that keeps track of attempted counts and success counts
    """
    def __init__(self, config):
        super().__init__(config)
        self.init_previous_task_stats()
    
    def init_previous_task_stats(self):
        self.task_attempted_counter = [0 for _ in range(len(self.possible_tasks))]
        self.task_success_counter = [0 for _ in range(len(self.possible_tasks))]

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
                task_id = self.possible_task_to_id[traj_task]
                with open(os.path.join(traj_path, "success.txt")) as f:
                    traj_success = f.read().strip().lower()
                self.task_attempted_counter[task_id] += 1
                self.task_success_counter[task_id] += traj_success == "true"
    
    def get_task_attempted_counter(self):
        return {
            self.possible_tasks[i]: self.task_attempted_counter[i] 
            for i in range(len(self.possible_tasks))
        }

    def log_task_completion(self, completion_info: dict):
        assert "success" in completion_info
        assert "task_str" in completion_info

        task_str = completion_info["task_str"]
        task_idx = self.possible_task_to_id[task_str]

        self.task_attempted_counter[task_idx] += 1
        self.task_success_counter[task_idx] += completion_info["success"] == True


class CyclingTaskProposer(ProposerWithCounters):
    """
    This proposer will cycle through the list of possible tasks
    """
    def __init__(self, config):
        super().__init__(config)
        self.current_task_idx = 0
    
    def propose_task(self, image_obs: np.ndarray, off_limits_objects=None):
        task = self.possible_tasks[self.current_task_idx]
        self.current_task_idx = (self.current_task_idx + 1) % len(self.possible_tasks)
        
        print("Proposing task:", task)
        
        return task


class VLMTaskProposer(ProposerWithCounters):
    def __init__(self, config):
        super().__init__(config)

        # Randomize task ordering
        random.seed(time.time())
        random.shuffle(self.possible_tasks)

        # For each task, convert task description into one or more VQA style questions that can be used to assess task affordance
        self.task_to_vqa_prompt = """To determine whether or not I should command the task, I have access to a vision-language model (VLM) that can answer true/false questions for me when I provide it an image of the lab environment the robot is operating in. Since I can only ask it true/false questions about the image, I need to convert the task description into a question that I can feed into the VLM.\n\nTo determine whether or not I should command the task, I have access to a vision-language model (VLM) that can answer true/false questions for me when I provide it an image of the lab environment the robot is operating in. Since I can only ask it true/false questions about the image, I need to convert the task description into a question that I can feed into the VLM.\n\nThe answer that the VLM returns (i.e., either true or false) may imply different things depending on the exact question that I asked it. In other words, the VLM returning "true" might mean I can command the task, or the VLM returning "false" might mean I can command the task (and the VLM returning "true" means I cannot command the task).\n\nHere are several examples:\n\nTask: put the yellow ball on the blue plate\nVLM question: Is the yellow ball currently on the blue plate?\nAnswer that implies task is feasible: false\n\nTask: move the yellow ball from the blue plate to the table\nVLM question: Is the yellow ball currently on the blue plate?\nAnswer that implies task is feasible: true\n\nTask: move the yellow ball to the left side of the table\nVLM question: Is the yellow ball on the right side of the table?\nAnswer that implies task is feasible: true\n\nTask: move the yellow ball to the right side of the table\nVLM question: Is the yellow ball on the left side of the table?\nAnswer that implies task is feasible: true\n\nTask: put the orange crayon on the blue plate\nVLM question: Is the orange crayon currently on the blue plate?\nAnswer that implies task is feasible: false\n\nTask: move the orange crayon from the blue plate to the table\nVLM question: Is the orange crayon currently on the blue plate?\nAnswer that implies task is feasible: true\n\nTask: put the orange crayon on the cloth\nVLM question: Is the orange crayon on top of the cloth?\nAnswer that implies task is feasible: false\n\nTask: move the orange crayon from the cloth to the table\nVLM question: Is the orange crayon on top of the cloth?\nAnswer that implies task is feasible: true\n\nTask: move the orange crayon from the cloth to the blue plate\nVLM question: Is the orange crayon on top of the cloth?\nAnswer that implies task is feasible: true\n\nTask: move the orange crayon from the blue plate to the cloth\nVLM question: Is the orange crayon on the blue plate?\nAnswer that implies task is feasible: true\n\nTask: move the orange crayon to the right side of the table\nVLM question: Is the orange crayon on the left side of the table?\nAnswer that implies task is feasible: true\n\nTask: move the orange crayon to the left side of the table\nVLM question: Is the orange crayon on the right side of the table?\nAnswer that implies task is feasible: true\n\nTask: move the red object from the blue plate to the table\nVLM question: Is the red object on the blue plate?\nAnswer that implies task is feasible: true\n\nTask: put the red object on the blue plate\nVLM question: Is the red object on the blue plate?\nAnswer that implies task is feasible: false\n\nTask: move the red object to the left side of the table\nVLM question: Is the red object on the right side of the table?\nAnswer that implies task is feasible: true\n\nTask: move the red object to the right side of the table\nVLM question: Is the red object on the left side of the table?\nAnswer that implies task is feasible: true\n\nTask: put the red object on the cloth\nVLM question: Is the red object on the cloth?\nAnswer that implies task is feasible: false\n\nTask: move the red object from the cloth to the table\nVLM question: Is the red object on the cloth?\nAnswer that implies task is feasible: true\n\nTask: move the red object from the cloth to the blue plate\nVLM question: Is the red object on the cloth?\nAnswer that implies task is feasible: true\n\nTask: move the red object from the blue plate to the cloth\nVLM question: Is the red object on the blue plate?\nAnswer that implies task is feasible: true\n\nTask: move the cloth to the right side of the table\nVLM question: Is the cloth on the left side of the table?\nAnswer that implies task is feasible: true\n\nTask: move the cloth to the left side of the table\nVLM question: Is the cloth on the right side of the table?\nAnswer that implies task is feasible: true\n\nExactly following the format of these examples, give me the VLM question and the answer that implies task is feasible for the following task:\n\nTask: """
        
        # Prompt to decode VLM output into true/false
        self.prompt_to_parse_vlm_output = [
            "I gave a vision-language model (VLM) an image and the following question: ",
            " In response, the VLM answered the following: ",
            " If the VLM answered in the affirmative, return just the word true. Otherwise if the VLM answered in the negative, return the word false."
        ]

        # cache for anything you want
        self.cache = {}

    def get_viable_tasks(self, image_obs: np.ndarray, off_limits_objects=None):
        if off_limits_objects is None:
            filtered_task_list = self.possible_tasks
        else:
            filtered_task_list = []
            for task in self.possible_tasks:
                off_limits = False
                for obj in off_limits_objects:
                    if obj in task:
                        off_limits = True
                        break
                if not off_limits:
                    filtered_task_list.append(task)
        
        task_to_vqa_prompts = [self.task_to_vqa_prompt + task for task in filtered_task_list]
        vqa_prompts = utils.ask_gpt4_batched(task_to_vqa_prompts, cache=self.cache)

        # add all the responses to the cache
        for q, r in zip(task_to_vqa_prompts, vqa_prompts):
            self.cache[q] = r

        vqa_prompts_parsed = []
        for vqa_prompt in vqa_prompts:
            vqa_prompt = vqa_prompt.strip()
            [part1, part2] = vqa_prompt.split("\n")
            vqa_prompts_parsed.append({
                "vlm_question" : part1[part1.index(":")+2:],
                "answer" : part2[part2.index(":")+2:]
            })
            print("vqa question", vqa_prompts_parsed[-1])

        images = [image_obs] * len(vqa_prompts_parsed)
        vlm_prompts = [vqa_prompts_parsed[i]["vlm_question"] for i in range(len(vqa_prompts_parsed))]
        if self.config["task_proposer_params"]["which_vlm"] == "gpt4v":
            vlm_answers = utils.ask_gpt4v_batched(images, vlm_prompts)
        elif self.config["task_proposer_params"]["which_vlm"] == "cogvlm":
            vlm_answers = utils.ask_cogvlm_batched(images, vlm_prompts, self.config)
        else:
            # If there's no VLM, all tasks are fair game
            viable_task_ids = []
            for i, task in enumerate(filtered_task_list):
                for j in range(len(self.possible_tasks)):
                    if self.possible_tasks[j] == task:
                        viable_task_ids.append(j)
                        break
            return viable_task_ids
        
        print("vlm_answers:", vlm_answers)

        decoding_prompts = [self.prompt_to_parse_vlm_output[0] + vlm_prompts[i] + self.prompt_to_parse_vlm_output[1] + vlm_answers[i] + self.prompt_to_parse_vlm_output[2] for i in range(len(vlm_answers))]
        print("decoding prompts:", decoding_prompts)
        vlm_answers_decoded = utils.ask_gpt4_batched(decoding_prompts, cache=self.cache)
        print("vlm_answers_decoded:", vlm_answers_decoded)

        # add all the responses to the cache
        for q, r in zip(decoding_prompts, vlm_answers_decoded):
            self.cache[q] = r

        vlm_answers_decoded = [vlm_answer.strip().lower() for vlm_answer in vlm_answers_decoded]
        task_viable_answers = [vqa_prompt["answer"].strip().lower() for vqa_prompt in vqa_prompts_parsed]
        viable_tasks_bool = (np.array(task_viable_answers) == np.array(vlm_answers_decoded)).tolist()
        viable_task_ids = []
        for i, task in enumerate(filtered_task_list):
            if viable_tasks_bool[i]:
                for j in range(len(self.possible_tasks)):
                    if self.possible_tasks[j] == task:
                        viable_task_ids.append(j)
                        break
                    
        return viable_task_ids

    def propose_task(self, image_obs: np.ndarray, off_limits_objects=None):
        task_ids = self.get_viable_tasks(image_obs, off_limits_objects)
        if len(task_ids) == 0: # handle case where no task is viable
            print("No viable tasks, selecting random task")
            task_ids = list(self.id_to_possible_task.keys())

        # With some probability we will ignore the VLM
        if random.random() < self.config["task_proposer_params"]["rand_selection_prob"]:
            return random.choice(self.possible_tasks)

        # We will sample uniformly from the remaining task options
        return self.possible_tasks[random.choice(task_ids)]

    def log_task_completion(self, completion_info: dict):
        pass


class HistoryAwareVLMTaskProposer(VLMTaskProposer):
    def __init__(self, config):
        super().__init__(config)
        
        self.zone_center = self.config["task_proposer_params"]["zone_center"]
        self.ucb_weight = self.config["task_proposer_params"]["ucb_weight"]

    def propose_task(self, image_obs: np.ndarray, off_limits_objects=None):
        task_ids = self.get_viable_tasks(image_obs, off_limits_objects)
        if len(task_ids) == 0: # handle case where no task is viable
            task_ids = [0]

        # We will use UCB to select tasks to command, where the Q-value of each 
        # task is a measure of distance to zone of proximal development
        ucb_scores = []
        total_task_executions = sum(self.task_attempted_counter)
        for task_id in task_ids:
            score = np.sqrt(np.log(total_task_executions+1) / (self.task_attempted_counter[task_id]+1))
            print(f"Task \"{self.possible_tasks[task_id]}\": UCB score: {score}, ", end="")
            task_success_rate = 0.0 if self.task_attempted_counter[task_id] == 0 else self.task_success_counter[task_id] / self.task_attempted_counter[task_id]
            dist_to_zone_score = 1.0 - math.fabs(task_success_rate - self.zone_center)
            print(f"dist to zone: {dist_to_zone_score}, ", end="")
            score = dist_to_zone_score + self.ucb_weight * score
            print(f"Q + UCB score: {score}")
            ucb_scores.append(score)
        ucb_scores = np.array(ucb_scores)
        argmax_idx = np.argmax(ucb_scores)

        # With some probability we will command a random task
        if random.random() < self.config["task_proposer_params"]["rand_selection_prob"]:
            print("Randomly selecting a task...")
            return random.choice(self.possible_tasks)

        return self.possible_tasks[task_ids[argmax_idx]]

    def log_task_completion(self, completion_info: dict):
        assert "success" in completion_info
        assert "task_str" in completion_info

        task_str = completion_info["task_str"]
        task_idx = -1
        for i in range(len(self.possible_tasks)):
            if self.possible_tasks[i] == task_str:
                task_idx = i
                break 
        assert task_idx != -1

        self.task_attempted_counter[task_idx] += 1
        self.task_success_counter[task_idx] += completion_info["success"] == True


class HumanTaskProposer(AbstractTaskProposer):
    """
    This will be a very simple (yet very intelligent) task 
    proposer that just queries the human
    """
    def __init__(self, config):
        super().__init__(config)
        self.old_prompt = None

    def propose_task(self, image_obs: np.ndarray, off_limits_objects):
        if self.old_prompt is not None:
            print("Previous prompt:", self.old_prompt)
            # if the user enters an empty string, we will use the previous prompt
        prompt = input("Enter prompt (press enter to use previous prompt): ")
        if prompt == "":
            prompt = self.old_prompt
        else:
            self.old_prompt = prompt
        return prompt
        
    def log_task_completion(self, completion_info: dict):
        pass


task_proposers = {
    "human": HumanTaskProposer,
    "none": HistoryAwareVLMTaskProposer,
    "cogvlm": HistoryAwareVLMTaskProposer,
    "gpt4v": HistoryAwareVLMTaskProposer,
    "cycling": CyclingTaskProposer,
}
