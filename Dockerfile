FROM python:3.11-slim

ENV HF_HOME=/app/.cache/huggingface \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000

WORKDIR /app

# CPU-only torch first (satisfies laya's torch>=2.0 requirement, no CUDA bloat),
# then laya + serve extras pinned to the locally verified combo.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "laya[serve]==0.3.20" "transformers==5.15.0"

COPY run_service.py /app/run_service.py

EXPOSE 8000

CMD ["python", "run_service.py"]