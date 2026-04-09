FROM docker.io/hiyouga/llamafactory:latest

WORKDIR /workspace/PHY-LLM-verifier

COPY . .

RUN python -m pip install -U pip setuptools wheel && \
    pip uninstall -y transformers huggingface-hub datasets fsspec deepspeed || true && \
    pip install \
      "transformers==4.52.4" \
      "datasets==3.6.0" \
      "huggingface-hub==0.36.2" \
      "fsspec[http]==2025.3.0" \
      "numpy>=1.26.0" \
      "pydantic>=2.7.0" \
      "pyyaml>=6.0.1" \
      "scikit-learn>=1.4.0" && \
    DS_BUILD_OPS=0 pip install "deepspeed==0.14.4"

ENV PYTHONPATH=/workspace/PHY-LLM-verifier
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["bash"]
