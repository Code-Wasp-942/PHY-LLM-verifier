FROM docker.io/hiyouga/llamafactory:latest

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install -U pip setuptools wheel && \
    pip uninstall llamafactory && \
    pip uninstall -y transformers huggingface-hub datasets fsspec deepspeed || true && \
    pip install -r /tmp/requirements.txt && \
    DS_BUILD_OPS=0 pip install "deepspeed>=0.14.4" && \
    python -m pip check

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["bash"]
