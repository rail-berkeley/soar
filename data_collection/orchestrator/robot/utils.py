import numpy as np
from pyquaternion import Quaternion
from PIL import Image
import os
import requests
from typing import List
from openai import OpenAI
import io
import base64
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

def state_to_eep(xyz_coor, zangle: float):
    """
    Implement the state to eep function.
    Refered to `bridge_data_robot`'s `widowx_controller/widowx_controller.py`
    return a 4x4 matrix
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(
        matrix=DEFAULT_ROTATION
    )
    new_pose[:3, :3] = new_quat.rotation_matrix
    # yaw, pitch, roll = quat.yaw_pitch_roll
    return new_pose


def get_observation(widowx_client, config):
    while True:
        obs = widowx_client.get_observation()
        if obs is None:
            print("WARNING: failed to get robot observation, retrying...")
        else:
            break
    obs["image"] = (
        obs["image"]
        .reshape(3, config["general_params"]["shoulder_camera_image_size"], config["general_params"]["shoulder_camera_image_size"])
        .transpose(1, 2, 0)
        * 255
    ).astype(np.uint8)
    return obs

def encode_image_np(image_np: np.ndarray):
    # Ensure the NumPy array is in uint8
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image_np)
    # Create a buffer to hold the bytes
    buffer = io.BytesIO()
    # Save the image to the buffer in PNG format (or JPEG or any other format)
    pil_image.save(buffer, format="PNG")
    # Encode the buffer's content in base64
    base64_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_encoded

def ask_gpt4v(image: np.ndarray, prompt: str):
    # We will first resize the image to 512x512
    if not image.shape == (512, 512, 3):
        image_pil = Image.fromarray(image)
        resized_pil = image_pil.resize((512, 512), Image.ANTIALIAS)
        image = np.array(resized_pil)

    # Prepare jsons for openai api requests
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }
    payload = {
        "model": "gpt-4-turbo", # gpt4o
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "###############"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "###############",
                            "detail": "low"
                        }
                    },
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.0
    }

    base64_image = encode_image_np(image)
    payload["messages"][0]["content"][1]["image_url"]["url"] = f"data:image/jpeg;base64,{base64_image}"
    payload["messages"][0]["content"][0]["text"] = prompt

    while True:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
        if "error" in response:
            continue
        assistant_message = response["choices"][0]["message"]["content"]
        break

    return assistant_message

def ask_gpt4v_batched(images: List[np.ndarray], prompts: List[str]):
    assert len(images) == len(prompts)

    # TODO: implement real batching
    assistant_messages = []
    print("querying gpt4v batched")
    for i in tqdm(range(len(images))):
        assistant_messages.append(ask_gpt4v(images[i], prompts[i]))

    return assistant_messages

def ask_gpt4(prompt, cache=None):
    if type(prompt) == tuple:
        prompt, cache = prompt
        
    # check if cache contains the answer
    if cache is not None and prompt in cache:
        return cache[prompt]
    
    # Prepare jsons for openai api requests
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }
    payload = {
        "model": "gpt-3.5-turbo-1106",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "###############"
                    },
                ]
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.0
    }

    payload["messages"][0]["content"][0]["text"] = prompt
    while True:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
        except:
            # sometime we get requests.exceptions.JSONDecodeError
            continue
        if "error" in response:
            continue
        assistant_message = response["choices"][0]["message"]["content"]
        break

    return assistant_message

def ask_gpt4_batched(prompts, cache=None):
    if prompts is None or len(prompts) == 0:
        return []
    if cache is not None:
        # zip cache with the prompts list
        prompts = [(prompt, cache) for prompt in prompts]
    with Pool(len(prompts)) as p:
        assistant_messages = p.map(ask_gpt4, prompts)
    return assistant_messages

#def ask_gpt4_batched(prompts):
#    assistant_messages = []
#    for prompt in tqdm(prompts):
#        assistant_messages.append(ask_gpt4(prompt))
#    return assistant_messages

def ask_cogvlm(image: np.ndarray, prompt: str, config):
    image_list, prompt_list = [image], [prompt]
    return ask_cogvlm_batched(image_list, prompt_list, config)[0]

def ask_cogvlm_batched(images: List[np.ndarray], prompts: List[str], config):

    def _ask_cogvlm_batched_helper():
        assert len(images) == len(prompts)

        files = []
        for i, (numpy_image, prompt) in enumerate(zip(images, prompts)):
            pil_image = Image.fromarray(numpy_image.astype('uint8'))
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')  # can be JPEG or other formats
            img_byte_arr.seek(0)

            # Append the image file
            files.append(('image', (f'image_{i}.png', img_byte_arr, 'image/png')))

            # Append the corresponding prompt
            files.append((f'prompt', (None, prompt)))

        url = "http://" + config["cogvlm_server_params"]["cogvlm_server_ip"] + ":" + str(config["cogvlm_server_params"]["cogvlm_server_port"]) + "/query"
        response = requests.post(url, files=files)

        return response.json()["response"]

    # repeat the query if it fails
    while True:
        try:
            response = _ask_cogvlm_batched_helper()
            break
        except:
            continue
    
    return response
