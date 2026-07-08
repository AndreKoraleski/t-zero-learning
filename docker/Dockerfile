FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /t-zero

RUN apt-get update && apt-get install --no-install-recommends ffmpeg libgl1 libegl1 libglfw3 libglew2.2 -y && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /work/requirements.txt
RUN pip install -r requirements.txt
ENV MUJOCO_GL=egl

VOLUME ["/t-zero"]