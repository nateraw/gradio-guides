from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from torch import nn

LABELS = Path(hf_hub_download('nateraw/quickdraw', 'class_names.txt')).read_text().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)
weights_file = hf_hub_download('nateraw/quickdraw', 'pytorch_model.bin')
state_dict = torch.load(weights_file, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()


def predict(im):
    x = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        out = model(x)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    values, indices = torch.topk(probabilities, 5)

    return {LABELS[i]: v.item() for i, v in zip(indices, values)}


interface = gr.Interface(predict, inputs='sketchpad', outputs='label', live=True)

if __name__ == '__main__':
    interface.launch(debug=True)
