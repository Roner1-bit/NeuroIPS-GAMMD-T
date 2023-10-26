FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

RUN pip install -U --no-cache-dir git+https://github.com/Roner1-bit/NeuroIPS-GAMMD-T.git

WORKDIR /workspace
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="Roner1/LLaMA-Colosal-NeuroIPS"

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
