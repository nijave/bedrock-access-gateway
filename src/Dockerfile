FROM public.ecr.aws/lambda/python:3.12

# Apply the latest OS security patches available at build time so rebuilt
# images pick up CVE fixes that have not yet rolled into the base image.
RUN dnf upgrade -y && dnf clean all

# Add Lambda Web Adapter for API Gateway response streaming
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

COPY ./api ./api

# Install the exact versions in uv.lock, with hashes checked, into the system
# Python. uv and the lock files are mounted for this step only.
RUN --mount=from=ghcr.io/astral-sh/uv:0.13.9,source=/uv,target=/bin/uv \
    --mount=type=bind,source=pyproject.toml,target=/tmp/uv-project/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/tmp/uv-project/uv.lock \
    uv export --project /tmp/uv-project --frozen --no-emit-project --no-dev -o /tmp/requirements.txt && \
    uv pip install --system --no-cache --require-hashes -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Preload tiktoken encoding: https://github.com/aws-samples/bedrock-access-gateway/issues/118
ENV TIKTOKEN_CACHE_DIR=/var/task/.cache/tiktoken
RUN python3 -c 'import tiktoken_ext.openai_public as tke; tke.cl100k_base()'

# Lambda Web Adapter requires overriding the Lambda base image entrypoint
# to run the web app directly instead of the Lambda runtime handler
ENTRYPOINT []
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]