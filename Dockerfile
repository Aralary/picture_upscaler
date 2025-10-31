# Базовый образ с PyTorch + CUDA 12.1
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Рабочая директория
WORKDIR /app

# Копируем проект в контейнер
COPY . .

# Устанавливаем системные зависимости (для OpenCV и git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple

# Исправляем проблему с torchvision/basicsr
RUN python fix_libs_docker.py

# Точка входа
ENTRYPOINT ["python", "main.py"]
