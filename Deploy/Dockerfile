# 1) Base image
FROM python:3.10-slim

# 2) Prevent .pyc creation & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3) System deps for OpenCV & ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libsm6 \
      libxext6 \
      gcc \
    && rm -rf /var/lib/apt/lists/*

# 4) Set working directory
WORKDIR /app

# 5) Copy requirements before code to maximize cache re-use
COPY requirements.txt .

# 6) Install CPU-only PyTorch & TorchVision
RUN pip install --no-cache-dir \
      torch==2.2.2+cpu torchvision==0.17.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# 7) Install the rest of Python deps
RUN pip install --no-cache-dir -r requirements.txt

# 8) Copy your application code
COPY app.py .

# 9) Expose Gradio port
EXPOSE 7860

# 10) Launch the Gradio app
CMD ["python", "app.py"]



