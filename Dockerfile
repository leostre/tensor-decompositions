FROM docker.io/pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel
WORKDIR /tdecomp
COPY pyproject.toml pyproject.toml
RUN uv pip sync --system --break-system-packages pyproject.toml --group=experiments
# ENTRYPOINT ["/bin/bash -c"]