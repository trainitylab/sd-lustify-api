FROM runpod/stable-diffusion:web-ui-10.2.1

WORKDIR /workspace/stable-diffusion-webui

RUN git clone https://github.com/Bing-su/adetailer.git extensions/adetailer && \
    git clone https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet

RUN mkdir -p models/Stable-diffusion && \
    curl -L -H "Authorization: Bearer 5a8a715385d19bc70872703503e89df5" \
    -o models/Stable-diffusion/lustify_sdxl.safetensors \
    "https://civitai.com/api/download/models/1569593?type=Model&format=SafeTensor&size=pruned&fp=fp16"

RUN mkdir -p models/ControlNet && \
    cd models/ControlNet && \
    wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin

RUN mkdir -p models/Lora && \
    cd models/Lora && \
    wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors

RUN mkdir -p models/ESRGAN && \
    cd models/ESRGAN && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

RUN pip install runpod insightface onnxruntime-gpu

COPY handler.py /workspace/handler.py

WORKDIR /workspace
EXPOSE 7860

CMD cd /workspace/stable-diffusion-webui && \
    python launch.py --listen --port 7860 --api --xformers --no-half-vae --skip-torch-cuda-test & \
    cd /workspace && python handler.py
