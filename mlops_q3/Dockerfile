FROM python:3.10-slim

WORKDIR /app

# System deps in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake wget git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package installation
RUN pip install uv --no-cache-dir

# ONNX Runtime C++ — download and install
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz \
    && tar -xzf onnxruntime-linux-x64-1.17.3.tgz \
    && cp -r onnxruntime-linux-x64-1.17.3/include/* /usr/local/include/ \
    && cp -r onnxruntime-linux-x64-1.17.3/lib/* /usr/local/lib/ \
    && ldconfig \
    && rm -rf onnxruntime-linux-x64-1.17.3*

# CPU-only torch first (heaviest, separate layer for better cache reuse)
RUN uv pip install --system --no-cache \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Remaining Python deps
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# C++ source files — copy and build before Python scripts
# (rebuilds only if .cpp or CMakeLists.txt changes)
COPY task4_cpp_inference.cpp CMakeLists.txt .
RUN cmake -S . -B build -DCMAKE_PREFIX_PATH=/usr/local \
    && cmake --build build --parallel $(nproc)

# Python scripts copied last — changes here won't invalidate above layers
COPY task1_train_lora.py \
     task2_export_onnx.py \
     task3_python_inference.py .

CMD ["bash", "-c", "\
    echo '=== Task 1: LoRA Fine-Tuning ===' && python train_lora_task_1.py && \
    echo '=== Task 2: ONNX Export ===' && python export_onnx_task_2.py && \
    echo '=== Task 3: Python Inference ===' && python python_inference_task_3.py && \
    echo '=== Task 4: C++ Inference ===' && ./build/sd_inference ./onnx_models/unet.onnx \
"]