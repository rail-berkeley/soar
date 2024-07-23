from flask import Flask, render_template, request, jsonify, send_file
import os
import portalocker

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

robot_video_feeds = {
    'feed0': {'observation': None, 'goal': None, 'status': {'commanded_task': 'N/A', 'subgoal': 0, 'timestep': 0, 'task_success': 'N/A'}},
    'feed1': {'observation': None, 'goal': None, 'status': {'commanded_task': 'N/A', 'subgoal': 0, 'timestep': 0, 'task_success': 'N/A'}},
    'feed2': {'observation': None, 'goal': None, 'status': {'commanded_task': 'N/A', 'subgoal': 0, 'timestep': 0, 'task_success': 'N/A'}},
    'feed3': {'observation': None, 'goal': None, 'status': {'commanded_task': 'N/A', 'subgoal': 0, 'timestep': 0, 'task_success': 'N/A'}},
    'feed4': {'observation': None, 'goal': None, 'status': {'commanded_task': 'N/A', 'subgoal': 0, 'timestep': 0, 'task_success': 'N/A'}},
    'feed5': {'observation': None, 'goal': None, 'status': {'commanded_task': 'N/A', 'subgoal': 0, 'timestep': 0, 'task_success': 'N/A'}},
    'feed6': {'observation': None, 'goal': None, 'status': {'commanded_task': 'N/A', 'subgoal': 0, 'timestep': 0, 'task_success': 'N/A'}},
    'feed7': {'observation': None, 'goal': None, 'status': {'commanded_task': 'N/A', 'subgoal': 0, 'timestep': 0, 'task_success': 'N/A'}}
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload/<robot_idx>', methods=['POST'])
def upload_image(robot_idx):
    robot_idx = int(robot_idx)
    image_type = request.args.get('type', '')
    if image_type not in ['observation', 'goal']:
        return jsonify({"error": "Invalid image type"}), 400

    file = request.files.get('file')
    if file and file.filename:
        # Save the file as 'observation.jpg' or 'goal.jpg' in the uploads folder
        filename = f"{image_type}_{robot_idx}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, 'wb') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            f.write(file.read())
            portalocker.unlock(f)
        return jsonify({"message": "Image uploaded successfully", "type": image_type, "robot_index" : robot_idx}), 200
    else:
        return jsonify({"error": "No file part"}), 400

@app.route('/images/<robot_idx>/<image_type>', methods=['GET'])
def get_latest_image(robot_idx, image_type):
    robot_idx = int(robot_idx)
    if image_type not in ['observation', 'goal']:
        return jsonify({"error": "Invalid image type"}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{image_type}_{robot_idx}.jpg")
    if os.path.exists(image_path):
        return send_file(image_path)
    else:
        return jsonify({"error": "Image not found"}), 404

@app.route('/update_status/<robot_idx>', methods=['POST'])
def update_status(robot_idx):
    data = request.get_json()
    robot_video_feeds["feed" + robot_idx]["status"] = {
        'commanded_task': data.get('commanded_task', ''),
        'subgoal': data.get('subgoal', 0),
        'timestep': data.get('timestep', 0),
        'task_success': data.get('task_success', ''),
    }
    return jsonify({'message': 'Status updated successfully'})

@app.route('/get_status_data/<robot_idx>')
def get_status_data(robot_idx):
    return jsonify(robot_video_feeds["feed" + robot_idx]["status"])

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)