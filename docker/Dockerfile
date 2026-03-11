# ----------------------------
# Stage 1: Builder for venv
# ----------------------------
    FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu22.04 AS builder

    ENV NVIDIA_VISIBLE_DEVICES=all
    ENV NVIDIA_DRIVER_CAPABILITIES=all
    
    
    # Install UV
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
    
    WORKDIR /workspace
    
    # Use hardlink for faster builds
    ENV UV_LINK_MODE=copy
    ENV UV_PROJECT_ENVIRONMENT=/.venv
    
    RUN apt-get update && apt-get install -y git ca-certificates && update-ca-certificates 
    
    
    # Copy only requirements for caching
    # COPY ./docker/requirements.txt /tmp/requirements.txt
    COPY ./pyproject.toml /tmp/pyproject.toml
    RUN touch /tmp/README.md
    
    # Create venv and install deps
    RUN uv venv --python 3.11 $UV_PROJECT_ENVIRONMENT 
    RUN uv sync --project /tmp/pyproject.toml --no-install-project

    
    # ----------------------------
    # Stage 2: Final runtime image
    # ----------------------------
    FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu22.04
    
    # Install OS dependencies in one RUN
    RUN apt-get update && apt-get install -y \
        ffmpeg \
        libgl1 vulkan-tools git \
        libgomp1 libxt6 libxmu6 libxrandr2 libxcursor1 libxinerama1 libxi6 \
        libxrender1 libxcomposite1 libxdamage1 libxfixes3 libxkbcommon0 libxkbcommon-x11-0 \
        libsm6 libice6 libglu1-mesa mesa-utils \
        && rm -rf /var/lib/apt/lists/*
    
    # Install UV
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
    
    ENV NVIDIA_VISIBLE_DEVICES=all
    ENV NVIDIA_DRIVER_CAPABILITIES=all
    ENV OMNI_KIT_ALLOW_ROOT=1
    ENV ACCEPT_EULA=Y
    WORKDIR /workspace
    
    # Copy the venv from builder
    COPY --from=builder /.venv /.venv
    ENV UV_LINK_MODE=hardlink
    ENV VIRTUAL_ENV=/.venv
    ENV PATH="/.venv/bin:$PATH"
    
    # Optional: you can COPY your code here if you want static code baked in
    RUN uv venv --python 3.11 $UV_PROJECT_ENVIRONMENT

    # Create EULA_ACCEPTED file with `yes`
    RUN echo "yes" > /.venv/lib/python3.11/site-packages/isaacsim/kit/EULA_ACCEPTED
    
    ENTRYPOINT []
    CMD ["/bin/bash"]