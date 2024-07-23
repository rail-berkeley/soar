import cv2
import requests
import threading
import time

# URLs for the POST endpoints
OBSERVATION_URL = 'http://localhost:5000/upload?type=observation'
GOAL_URL = 'http://localhost:5000/upload?type=goal'

# Paths to the video files
OBSERVATION_VIDEO_PATH = 'trajectory.mp4'
GOAL_VIDEO_PATH = 'goals.mp4'

def send_frames(video_path, url, frame_interval=1):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Video ended or error, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            continue
            
        if frame_count % frame_interval == 0:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Could not encode image, skipping frame...")
                frame_count += 1
                continue
            
            files = {'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
            response = requests.post(url, files=files)
            if response.status_code != 200:
                print(f"Failed to send frame: {response.content}")
        
        frame_count += 1
        time.sleep(0.1)

    cap.release()

def send_observation_frames():
    send_frames(OBSERVATION_VIDEO_PATH, OBSERVATION_URL, frame_interval=1)

def send_goal_frames():
    send_frames(GOAL_VIDEO_PATH, GOAL_URL, frame_interval=1)

if __name__ == '__main__':
    observation_thread = threading.Thread(target=send_observation_frames)
    goal_thread = threading.Thread(target=send_goal_frames)
    
    observation_thread.start()
    goal_thread.start()

    observation_thread.join()
    goal_thread.join()