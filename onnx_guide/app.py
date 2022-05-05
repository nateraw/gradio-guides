import json
import math

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
from huggingface_hub import hf_hub_download

modele = hf_hub_download(repo_id="onnx/EfficientNet-Lite4", filename="efficientnet-lite4-11.onnx")
# load the labels text file
labels = json.load(open("onnx_guide/labels_map.txt", "r"))

# set image file dimensions to 224x224 by resizing and cropping image from center
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img


# resize the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100.0 * out_height / scale)
    new_width = int(100.0 * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


# crop the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


sess = rt.InferenceSession(modele)


def inference(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = pre_process_edgetpu(img, (224, 224, 3))

    img_batch = np.expand_dims(img, axis=0)

    results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]
    result = reversed(results[0].argsort()[-5:])
    resultdic = {}
    for r in result:
        resultdic[labels[str(r)]] = float(results[0][r])
    return resultdic


title = "EfficientNet-Lite4"
description = "EfficientNet-Lite 4 is the largest variant and most accurate of the set of EfficientNet-Lite model. It is an integer-only quantized model that produces the highest accuracy of all of the EfficientNet models. It achieves 80.4% ImageNet top-1 accuracy, while still running in real-time (e.g. 30ms/image) on a Pixel 4 CPU."
examples = [[hf_hub_download('nateraw/gradio-guides-files', 'catonnx.jpg', repo_type='dataset', force_filename='catonnx.jpg')]]

interface = gr.Interface(
    inference, gr.inputs.Image(type="filepath"), "label", title=title, description=description, examples=examples
)

if __name__ == '__main__':
    interface.launch(debug=True)
