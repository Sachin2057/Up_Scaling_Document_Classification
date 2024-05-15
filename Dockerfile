
FROM nvidia/cuda:11.5.2-cudnn8-runtime-ubuntu20.04



WORKDIR /app
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    ffmpeg libsm6 libxext6  -y \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*
    RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py
COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118




COPY . /app


CMD ["streamlit", "run","web_app.py"]