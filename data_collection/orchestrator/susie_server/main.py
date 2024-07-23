import yaml
from yamlinclude import YamlIncludeConstructor
import argparse
import os
import numpy as np
from susie.model import create_sample_fn
from flask import Flask, request, send_file
from PIL import Image
import io

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', required=True)
args = parser.parse_args()

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=args.config_dir)
with open(os.path.join(args.config_dir, "config.yaml")) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class SubgoalPredictor:
    def __init__(self, config):
        diffusion_config = config["subgoal_predictor_params"]
        self.diffusion_sample_func = create_sample_fn(
            diffusion_config["diffusion_checkpoint"],
            diffusion_config["diffusion_wandb"],
            diffusion_config["diffusion_num_steps"],
            diffusion_config["prompt_w"],
            diffusion_config["context_w"],
            0.0,
            diffusion_config["diffusion_pretrained_path"],
        )
        self.image_size = diffusion_config["image_size"]

    def __call__(self, image_obs: np.ndarray, prompt: str):
        assert image_obs.shape == (
            self.image_size,
            self.image_size,
            3,
        ), "Bad input image shape"
        return self.diffusion_sample_func(image_obs, prompt)
    
subgoal_predictor = SubgoalPredictor(config)

app = Flask(__name__)

@app.route('/generate_subgoal', methods=['POST'])
def process_image():
    global subgoal_predictor

    # Check if the request contains the 'image' file
    if 'image' not in request.files:
        return "No image part", 400
    file = request.files['image']
    
    # Check if the request contains the 'text' part
    if 'text' not in request.form:
        return "No text provided", 400
    text = request.form['text']
    
    # Read the image file
    image = Image.open(file.stream)
    image = np.array(image)

    generated_subgoal = subgoal_predictor(image_obs=image, prompt=text)
    
    # Save the image to a binary stream
    generated_subgoal = Image.fromarray(generated_subgoal)
    img_io = io.BytesIO()
    generated_subgoal.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7000)