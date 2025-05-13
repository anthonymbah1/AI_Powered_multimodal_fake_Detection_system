# 🧠 AI Powered Multimodal Deepfake Detection System

A capstone project developed to detect fake content across multiple modalities — **text, image, and video** — using fine-tuned deep learning models. This system is designed with the potential to be scaled into a real-world solution for identifying disinformation across various digital platforms.

---

## 🚀 Project Overview

This project leverages:
- **BERT** for disinformation detection in captions
- **ViT (Vision Transformer)** for image-based deepfake classification
- **ResNet50** for frame-level video classification

Models are developed and evaluated using Jupyter Notebooks on Google Colab. Final deployment is handled through a Gradio interface that fuses predictions from all three modalities.

---

## 🧱 Project Structure

```
AI_Powered_multimodal_Deepfake_Detection_system/
├── Deploy/
│   ├── fused_final_gradio.ipynb   # 🔥 Main interface to launch the app
│   ├── app.py                     # Gradio app logic
│   ├── Dockerfile                 # Optional: containerization setup
│   └── requirements.txt           # Deployment dependencies
├── Models_ipynb/
│   ├── *.ipynb                    # Training notebooks for BERT, ViT, ResNet50
├── .env.example                   # Sample environment config (AWS keys, S3 paths)
├── .gitignore
└── README.md                      # 📄 You're here
```

## 📦 Tech Stack
🧠 Transformers (bert-base-uncased)

- 🖼️ Vision Transformer (ViT) via Hugging Face

- 🎥 ResNet50 for frame-based video analysis

- ⚙️ Gradio for deployment

- ☁️ AWS S3 for cloud-based data storage and retrieval

- 🐍 Python, PyTorch, Docker

## 🛠️ How to Run
1. Clone the repository:
```
git clone https://github.com/anthonymbah1/AI_Powered_multimodal_Deepfake_Detection_system.git
cd AI_Powered_multimodal_Deepfake_Detection_system
```

2. Set up your environment:

- Create a .env file using .env.example

- Install dependencies using Deploy/requirements.txt

3. Launch the system:

- Open and run **Deploy/fused_final_gradio.ipynb** in Google Colab

## 📊 Results & Performance
- Multimodal fusion improves robustness and reduces false positives

- Preliminary tests show ~92% accuracy across combined modalities

## 🔐 Environment Variables
```
Add these to your .env file:
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
S3_TRAIN_KEY=csvfils/train.csv
S3_VALIDATION_KEY=csvfils/validation.csv
S3_TEST_KEY=csvfils/test.csv
```

## 👨🏽‍💻 Author
**Tony Mbah**
🔗 [LinkedIn](https://www.linkedin.com/in/tony-mbah)
