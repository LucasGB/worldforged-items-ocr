FROM public.ecr.aws/lambda/python:3.12

# Set environment variables early
ENV TMPDIR=/tmp
ENV PADDLEX_CACHE_DIR=/tmp/paddlex_cache
ENV HOME=/tmp

# Install system dependencies
RUN dnf install -y \
    gcc \
    gcc-c++ \
    cmake \
    make \
    wget \
    unzip \
    libgomp \
    mesa-libGL \
    && dnf clean all

# Install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ${LAMBDA_TASK_ROOT}/
COPY ocr_config.yaml ${LAMBDA_TASK_ROOT}/

# Pre-download PaddleOCR models
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

CMD ["lambda_handler.lambda_handler"]
