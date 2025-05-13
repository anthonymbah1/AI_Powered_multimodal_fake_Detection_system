import os
import csv
from datetime import datetime
import boto3
import gradio as gr
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import BertForSequenceClassification, BertTokenizer
from PIL import Image
import numpy as np

# 1. S3 model download utility
def download_model_from_s3(bucket, prefix, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'):
                continue
            local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)

# 2. Model paths
BUCKET_NAME = "staightouttathere-deepfake-projet"
MODELS_PREFIX = "models/"

local_model_root = "./models/"
os.makedirs(local_model_root, exist_ok=True)

# Download models only once
if not os.path.exists(os.path.join(local_model_root, "bert-original-caption-model")):
    download_model_from_s3(BUCKET_NAME, MODELS_PREFIX, local_model_root)

# 3. Load models
vit_model = ViTForImageClassification.from_pretrained(os.path.join(local_model_root, "vit_deepfake_model")).to('cpu')
vit_processor = ViTImageProcessor.from_pretrained(os.path.join(local_model_root, "vit_deepfake_model"))

bert_original_model = BertForSequenceClassification.from_pretrained(os.path.join(local_model_root, "bert-original-caption-model")).to('cpu')
bert_original_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_generated_model = BertForSequenceClassification.from_pretrained(os.path.join(local_model_root, "bert-generated-caption-model")).to('cpu')
bert_generated_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

vit_model.eval()
bert_original_model.eval()
bert_generated_model.eval()

# 4. CSV logging function for Power BI
def log_prediction(input_type, label, confidence):
    log_file = "predictions.csv"
    header = ['timestamp', 'input_type', 'label', 'confidence']

    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([datetime.now().isoformat(), input_type, label, f"{confidence:.2f}"])

# 5. Prediction Functions

def predict_image(image):
    inputs = vit_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probs, dim=1)

    label_map = {0: "Fake", 1: "Real"}
    label = label_map[prediction.item()]
    confidence_percent = confidence.item() * 100

    # 🔥 Log prediction
    log_prediction("image", label, confidence_percent)

    return f"Image Prediction: {label} ({confidence_percent:.2f}%)"

def predict_text_original(text):
    inputs = bert_original_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_original_model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probs, dim=1)

    label_map = {0: "Fake News", 1: "Real News"}
    label = label_map[prediction.item()]
    confidence_percent = confidence.item() * 100

    # 🔥 Log prediction
    log_prediction("text-original", label, confidence_percent)

    return f"Original Caption Prediction: {label} ({confidence_percent:.2f}%)"

def predict_text_generated(text):
    inputs = bert_generated_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_generated_model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probs, dim=1)

    label_map = {0: "Fake News", 1: "Real News"}
    label = label_map[prediction.item()]
    confidence_percent = confidence.item() * 100

    # 🔥 Log prediction
    log_prediction("text-generated", label, confidence_percent)

    return f"Generated Caption Prediction: {label} ({confidence_percent:.2f}%)"

# 6. Gradio Interfaces

image_interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Deepfake Image Detector"
)

original_caption_interface = gr.Interface(
    fn=predict_text_original,
    inputs=gr.Textbox(lines=5, placeholder="Paste original caption text here..."),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Fake News Detection (Original Captions)"
)

generated_caption_interface = gr.Interface(
    fn=predict_text_generated,
    inputs=gr.Textbox(lines=5, placeholder="Paste generated caption text here..."),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Fake News Detection (Generated Captions)"
)

# 7. Group into tabs and launch
gr.TabbedInterface(
    [image_interface, original_caption_interface, generated_caption_interface],
    tab_names=["Detect Deepfake Image", "Detect Fake News (Original Captions)", "Detect Fake News (Generated Captions)"]
).launch()

