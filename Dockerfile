FROM mambaorg/micromamba:2.0.5

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment-container.yml /tmp/environment.yml
RUN micromamba install --yes --name base --file /tmp/environment.yml \
    && micromamba clean --all --yes

# Make the base Conda environment active during subsequent Docker build RUNs.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /opt/molearn
COPY --chown=$MAMBA_USER:$MAMBA_USER setup.py README.md ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src
RUN python -m pip install --no-deps .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-c", "import molearn, torch; print('molearn:', molearn.__file__); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"]
