FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu
# SDK wheel is distributed by Rockchip; copy it from the pinned SDK checkout.
COPY packages/ /sdk/packages/
RUN sed '/auto_gptq/d' /sdk/packages/requirements.txt >/tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && BUILD_CUDA_EXT=0 pip install --no-build-isolation --no-cache-dir auto_gptq==0.7.1 \
    && pip install --no-deps /sdk/packages/rkllm_toolkit-1.3.1-cp312-cp312-linux_x86_64.whl
WORKDIR /work
ENV PYTHONPATH=/work/src
ENTRYPOINT ["python", "scripts/export_rkllm.py"]
