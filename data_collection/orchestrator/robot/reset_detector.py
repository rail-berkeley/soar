import numpy as np
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import utils
from nltk.corpus import cmudict


def send_slack_message(client, channel_id, message):
    try:
        response = client.chat_postMessage(channel=channel_id, text=message)
        assert response["message"]["text"] == message
    except SlackApiError as e:
        assert e.response["ok"] is False
        # str like 'invalid_auth', 'channel_not_found'
        assert e.response["error"]
        print(f"Got an error: {e.response['error']}")


class ResetDetector:
    """
    Uses a VLM to predict whether an object is missing (and thus a reset is required).
    If so, the Slack API is used to send a notification
    """
    def __init__(self, config):
        self.config = config

        # Load object list
        self.objects = self.config["task_definition_params"]["object_list"]

        # Prepare prompt strings
        self.prompt_components = [
            "Is there ",
            " present in the image?",
            "I gave a vision-language model an image and asked if there was ",
            " present in the image. The following was the model's response:\n\n######\n",
            "\n######\n\nIf the model thought the object in question was present, return just the word 'true'. Else return 'false'."
        ]
        
        # connecting to slack
        slack_token = self.config["reset_detector_params"]["slack_token"]
        self.slack_client = WebClient(token=slack_token)

        # cache for anything you want
        self.cache = {}
        
        # cache the status of each object so we don't send slack messages every time
        self.cache["object_status"] = {
            obj: True 
            for obj in self.objects
        }

    def prepend_article(self, object_name):
        first_word = object_name.split()[0]
        def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
            for syllables in pronunciations.get(word, []):
                return syllables[0][-1].isdigit()  # use only the first one
        if starts_with_vowel_sound(first_word):
            return "an " + object_name
        return "a " + object_name

    def detect(self, image: np.ndarray):
        objects_present = []
        if self.config["reset_detector_params"]["which_vlm"] != "none":
            
            # prompts for the VLM for each object
            image_prompts = []
            for object in self.objects:
                image_prompt = self.prompt_components[0] + self.prepend_article(object) + self.prompt_components[1]
                image_prompts.append(image_prompt)
            
            # image for the VLM for each object    
            images = [image] * len(self.objects)
            
            if self.config["reset_detector_params"]["which_vlm"] == "cogvlm":
                vlm_output = utils.ask_cogvlm_batched(images, image_prompts, self.config)
            elif self.config["reset_detector_params"]["which_vlm"] == "gpt4v":
                vlm_output = utils.ask_gpt4v_batched(images, image_prompts)
            else:
                vlm_output = "Yes, " + self.prepend_article(object) + " is present in the image."
            print("vlm output:", vlm_output)
            
            # decoding prompts for the LLM
            decoding_prompts = [
                self.prompt_components[2] + \
                self.prepend_article(self.objects[i]) + \
                self.prompt_components[3] + \
                vlm_output[i].strip() + \
                self.prompt_components[4] 
                for i in range(len(self.objects))
            ]
            print("decoding prompts:", decoding_prompts)
            
            llm_output = utils.ask_gpt4_batched(decoding_prompts, cache=self.cache)
            for q, r in zip(decoding_prompts, llm_output):
                self.cache[q] = r
            
            # object presence info
            updated_object_status = {}
            for llm_answer, obj in zip(llm_output, self.objects):
                in_scene = llm_answer.strip().lower() == "true"
                try:
                    assert in_scene in (True, False)
                except AssertionError:
                    in_scene = False
                updated_object_status[obj] = in_scene
                print(f"Reset Detector LLM said {obj} is present:", in_scene)
                objects_present.append(in_scene)

        else:
            objects_present = [True] * len(self.objects)
            updated_object_status = {
                obj: True
                for obj in self.objects
            }

        objects_not_present = []
        for i in range(len(objects_present)):
            if not objects_present[i]:
                objects_not_present.append(self.objects[i])

        # missing objects are those in objects_not_present, but have previous status as True
        # i.e. they were present the last time ResetDetector was run but not this time
        missing_objects = [obj for obj in objects_not_present if self.cache["object_status"][obj]]
        if len(missing_objects) != 0:
            # Reset required, send slack message
            message = f"Hey! Robot {self.config['general_params']['robot_id']} is missing objects: "
            message += ", ".join(missing_objects)
            
            channel_id = self.config["reset_detector_params"]["channel_id"]
            send_slack_message(self.slack_client, channel_id, message)
        
        # update object status
        self.cache["object_status"] = updated_object_status

        # Prepare return dictionary
        to_return = {}
        for i in range(len(objects_present)):
            to_return[self.objects[i]] = objects_present[i] # we're commenting this out bc the VLM can be unreliable, and we don't want to not propose tasks we can actually execute
        return to_return

        
