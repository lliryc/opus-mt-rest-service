FROM python:3.10.11-slim-bullseye

WORKDIR /workspace

RUN pip3 install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python -c 'from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en"); model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ar-en")'

COPY static static
COPY opus_mt.py opus_mt.py
CMD ["uvicorn", "opus_mt:app", "--host", "0.0.0.0", "--port", "80"]
