FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple

RUN python fix_scripts/fix_libs_docker.py

ENTRYPOINT ["python", "main.py"]
