import torch
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import json
import numpy as np
from tqdm import tqdm
from flask import Flask
from flask import Flask, request, jsonify
import copy
import io

MODEL_PATH = "THUDM/cogvlm-chat-hf"
TOKENIZER_PATH = "lmsys/vicuna-7b-v1.5"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
torch_type = torch.bfloat16

print("Loading model to GPU mem")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_type,
    load_in_4bit=False,
    trust_remote_code=True
).to(DEVICE).eval()

app = Flask(__name__)

@app.route("/query", methods=['POST'])
def query_model():
    global tokenizer, model, torch_type, DEVICE

    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Missing image or prompt"}), 400
    
    prompt = request.form['prompt']

    file = request.files['image']
    image = Image.open(file.stream)

    # Resize image to 490x490
    image = np.array(image)[:, :, :3]
    image = Image.fromarray(image).resize((490, 490), Image.LANCZOS)

    input_by_model = model.build_conversation_input_ids(tokenizer, query=prompt, history=[], images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
    }

    gen_kwargs = {"max_length": 2048,
                      "temperature": 0.0,
                      "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("</s>")[0]
    
    return jsonify({
        "response": response,
    }), 200

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
