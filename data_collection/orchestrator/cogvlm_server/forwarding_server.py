import argparse
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from flask import Flask, request, jsonify
import copy
import io
from io import BytesIO
import requests

app = Flask(__name__)

@app.route("/query", methods=['POST'])
def forward_request():
    print("recieved request")
    image_files = request.files.getlist('image')
    prompts = request.form.getlist('prompt')
    if not image_files or not prompts or len(image_files) != len(prompts):
        return jsonify({"error": "Missing image(s) or prompt(s)"}), 400
    numpy_images, prompt_strings = [], []
    for i, file in enumerate(image_files):
        image = Image.open(file.stream)
        image = np.array(image)[:, :, :3]
        numpy_images.append(image)
        prompt_strings.append(prompts[i])
    files = []
    for i, (numpy_image, prompt) in enumerate(zip(numpy_images, prompt_strings)):
        pil_image = Image.fromarray(numpy_image.astype('uint8'))
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')  # can be JPEG or other formats
        img_byte_arr.seek(0)

        # Append the image file
        files.append(('image', (f'image_{i}.png', img_byte_arr, 'image/png')))

        # Append the corresponding prompt
        files.append((f'prompt', (None, prompt)))
    url = "http://localhost:7000/query"
    response = requests.post(url, files=files)
    if response.status_code == 200:
        json_response = response.json()
        return jsonify(json_response), 200
    else:
        return jsonify({"error": "something went wrong"}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=6000)
