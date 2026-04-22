FROM python:3.10-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache \
    "transformers==4.40.0" \
    "sentencepiece==0.2.0" \
    "sacrebleu==2.4.0" \
    "numpy<2.0"

COPY translate.py evaluate.py input.txt reference.txt .

CMD ["sh", "-c", "python translate.py && python evaluate.py"]