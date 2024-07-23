import rospy
from sensor_msgs.msg import Image
import time
import numpy as np
from PIL import Image as PILImage
import requests
import io
import yaml
from yamlinclude import YamlIncludeConstructor
from absl import app, flags
import os

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config_dir",
    None,
    "Path to config directory",
    required=True,
)

def run_ros_subscriber(_):
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=FLAGS.config_dir)
    with open(os.path.join(FLAGS.config_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    rospy.init_node('image_listener', anonymous=True)

    # Function to handle the image message once received
    def image_callback(message):
        rospy.loginfo("Received an image!")
        str_message = str(message)
        image_data = str_message[str_message.index("[")+1 : str_message.index("]")]
        image_data = image_data.replace(",", "").split()
        image_data = [int(elem) for elem in image_data]
        image = np.array(image_data, dtype=np.uint8)

        # The shape of the image is 480 x 640
        image = image.reshape(480, 640, 3)
        image = image[:, :, ::-1] # convert from BGR to RGB

        img = PILImage.fromarray(image)
        img = img.resize((512, 512), PILImage.LANCZOS) # make image square 512x512

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        files = {"file": ("image.jpg", buffer.getvalue(), "image/jpeg")}

        # We're sending this to our main web server
        url = 'http://' + config["general_params"]["web_viewer_ip"] + ':' + str(config["general_params"]["web_viewer_port"]) + '/upload/' + str(config["general_params"]["robot_id"]) + '?type=observation'
        response = requests.post(url, files=files)

    rospy.Subscriber('/blue/image_raw', Image, image_callback)

    rospy.spin()  # Spin until the node is shut down

if __name__ == "__main__":
    app.run(run_ros_subscriber)
