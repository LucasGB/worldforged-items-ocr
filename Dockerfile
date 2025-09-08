FROM public.ecr.aws/lambda/python:3.12

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

# Create models directory and pre-download PaddleOCR models


# Pre-download models by initializing PaddleOCR
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"


# RUN mkdir -p /tmp/paddlex_models
# ENV PADDLEX_HOME=/tmp/paddlex_models
# RUN python -c "\
# import os; \
# os.environ['PADDLEX_HOME'] = '/tmp/paddlex_models'; \
# from paddleocr import PaddleOCR; \
# print('Pre-downloading models...'); \
# ocr = PaddleOCR(use_angle_cls=True, lang='en'); \
# print('Models downloaded successfully'); \
# "

# Set the CMD to your handler
CMD ["lambda_handler.lambda_handler"]