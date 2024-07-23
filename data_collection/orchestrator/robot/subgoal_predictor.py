import numpy as np
import requests
from PIL import Image
import io

class SubgoalPredictor:
    def __init__(self, config):
        diffusion_config = config["subgoal_predictor_params"]
        self.image_size = diffusion_config["image_size"]
        self.url = "http://" + diffusion_config["susie_server_ip"] + ":" + str(diffusion_config["susie_server_port"]) + "/generate_subgoal"

    def numpy_to_image(self, np_array):
        """Convert a NumPy array to a PIL Image."""
        return Image.fromarray(np.uint8(np_array))

    def image_to_numpy(self, image):
        """Convert a PIL Image to a NumPy array."""
        return np.array(image)

    def send_image_and_text(self, url, np_image, text):
        """Send a NumPy image array and text to the specified URL."""
        # Convert NumPy array to PIL Image
        image = self.numpy_to_image(np_image)
        
        # Save the PIL Image to a bytes buffer
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Prepare files and data for the request
        files = {'image': ('image.jpg', img_buffer, 'image/jpeg')}
        data = {'text': text}
        
        # Send POST request
        response = requests.post(url, files=files, data=data)
        return response

    def __call__(self, image_obs: np.ndarray, prompt: str):
        assert image_obs.shape == (
            self.image_size,
            self.image_size,
            3,
        ), "Bad input image shape"

        response = self.send_image_and_text(self.url, image_obs, prompt)
        if response.status_code == 200:
            # Convert the response content back to a NumPy array
            image = Image.open(io.BytesIO(response.content))
            output_np_image = self.image_to_numpy(image)
        else:
            print("Failed to process image", response.status_code, response.text)
            return None
        
        return output_np_image