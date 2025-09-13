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
COPY ocr_config_prod.yaml ${LAMBDA_TASK_ROOT}/ocr_config.yaml

# Copy pre-downloaded models from your local project
RUN mkdir -p /opt/models 
COPY models/ /opt/models/

# Set the CMD to your handler
CMD ["lambda_handler.lambda_handler"]