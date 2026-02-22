# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
# Since I don't have cuda on my system, I cannot use above command.
FROM python:3.10-slim 

# 2. Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workdir

# Install essential system tools 'git' for cloning
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workdir/
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the newly created modules
COPY train.py eval.py /workdir/

# Run evaluation as instructed in Task 9. I ran the training script on Colab and pushed the model to Hugging Face.
CMD ["python", "eval.py"]