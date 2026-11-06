FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg espeak-ng libespeak-ng1 libgl1 libglib2.0-0 libsndfile1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

COPY author-requirements.txt constraints.txt /opt/pia-setup/
RUN python -m pip install --no-cache-dir pip==25.2 setuptools==75.8.0 wheel==0.45.1 \
    cython==3.0.12 numpy==1.26.4 \
    && python -m pip install --no-cache-dir --report /opt/torch-install.json \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir --no-build-isolation \
    --report /opt/pia-install.json -c /opt/pia-setup/constraints.txt \
    -r /opt/pia-setup/author-requirements.txt \
    && python -m pip check

COPY probe.py /opt/pia-probe.py
WORKDIR /pia
