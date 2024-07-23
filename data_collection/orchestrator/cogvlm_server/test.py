import requests
from pathlib import Path

# Your Flask API endpoint URL
url = 'http://localhost:5001/query'  # We're testing out ssh port forwarding

image_path = 'test-img.png'
prompt_text = 'Describe the image.'

# Make sure the image path is valid
if not Path(image_path).is_file():
    print("The image file does not exist.")
    exit()

# Open the image in binary mode
with open(image_path, 'rb') as image_file:
    # Prepare the data for the POST request
    payload = {
        'prompt': (None, prompt_text)
    }
    files = {
        'image': (image_path, image_file, 'multipart/form-data')
    }

    # Send the POST request to the Flask API endpoint
    response = requests.post(url, files=files, data=payload)

    # Check the response status code
    if response.ok:
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)