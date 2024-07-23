import requests
from PIL import Image
import io
import numpy as np

img = Image.open("test-img.png")

# Your images as numpy arrays
numpy_images = [np.array(img), np.array(img)]
prompts = ["On which side of the metal tray is the coke can?", "On which side of the metal tray is the capsicum?"]

numpy_images = 5 * numpy_images
prompts = 5 * prompts

# The server endpoint
url = "http://localhost:6000/query"

files = []
for i, (numpy_image, prompt) in enumerate(zip(numpy_images, prompts)):
    pil_image = Image.fromarray(numpy_image.astype('uint8'))
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')  # can be JPEG or other formats
    img_byte_arr.seek(0)

    # Append the image file
    files.append(('image', (f'image_{i}.png', img_byte_arr, 'image/png')))

    # Append the corresponding prompt
    files.append((f'prompt', (None, prompt)))

# Perform the request
response = requests.post(url, files=files)

# Response handling
if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Error:", response.status_code, response.text)

