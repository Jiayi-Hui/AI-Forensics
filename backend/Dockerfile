FROM python:3.10-slim

WORKDIR /app

# ffmpeg (video), libsndfile (audio), build tools (librosa C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (cached layer)
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# App code and assets (only what's needed at runtime)
COPY backend/        ./backend/
COPY 8307demo_huijiayi/ ./8307demo_huijiayi/
COPY modules/speech_detection/ ./modules/speech_detection/
COPY modules/text_detection/   ./modules/text_detection/

# Only copy the primary inference weight (saves ~490 MB vs. copying all three)
COPY weights/efficientnet_b7_deepfake.pth ./weights/efficientnet_b7_deepfake.pth

EXPOSE 8000

WORKDIR /app/backend
CMD ["python", "main.py"]
