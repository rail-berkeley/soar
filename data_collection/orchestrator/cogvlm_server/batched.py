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

    image_files = request.files.getlist('image')
    prompts = request.form.getlist('prompt')

    if not image_files or not prompts or len(image_files) != len(prompts):
        return jsonify({"error": "Missing image(s) or prompt(s)"}), 400

    model_inputs = []
    max_len = -1
    for i, file in enumerate(image_files):
        image = Image.open(file.stream)
        image = np.array(image)[:, :, :3]
        image = Image.fromarray(image).resize((490, 490), Image.LANCZOS)

        input_by_model = model.build_conversation_input_ids(tokenizer, query=prompts[i], history=[], images=[image])
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
        }
        model_inputs.append(inputs)
        max_len = max(max_len, inputs["input_ids"].shape[1])
    concatenated_images = []
    for i in range(len(model_inputs)):
        tensor_shape = model_inputs[i]["input_ids"].shape[1]
        model_inputs[i]["input_ids"] = torch.cat([torch.zeros((1, max_len-tensor_shape), dtype=torch.long, device=DEVICE), model_inputs[i]["input_ids"]], dim=1)
        model_inputs[i]["token_type_ids"] = torch.cat([torch.zeros((1, max_len-tensor_shape), dtype=torch.long, device=DEVICE), model_inputs[i]["token_type_ids"]], dim=1)
        model_inputs[i]["attention_mask"] = torch.cat([torch.zeros((1, max_len-tensor_shape), dtype=torch.long, device=DEVICE), model_inputs[i]["attention_mask"]], dim=1)
        concatenated_images.append(model_inputs[i]["images"][0])
    combined_inputs = {
        'input_ids': torch.cat([inputs['input_ids'] for inputs in model_inputs], dim=0),
        'token_type_ids': torch.cat([inputs['token_type_ids'] for inputs in model_inputs], dim=0),
        'attention_mask': torch.cat([inputs['attention_mask'] for inputs in model_inputs], dim=0),
        'images': concatenated_images
    }

    gen_kwargs = {"max_length": 2048,
                      "temperature": 0.0,
                      "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**combined_inputs, **gen_kwargs)
        generation_strings = []
        for i in range(len(outputs)):
            output_tokens = outputs[i][max_len:]
            generation = tokenizer.decode(output_tokens)
            generation = generation[:generation.index("</s>")]
            generation_strings.append(generation)

    return jsonify({
        "response": generation_strings,
    }), 200

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)

