# Aphrodite Engine Development Container
# Optimized for Deep Tree Echo architecture with multi-device support

# Use official Python dev containers base with CUDA support
FROM mcr.microsoft.com/devcontainers/python:3.12-bullseye as base

# Install system dependencies for Aphrodite Engine and Deep Tree Echo
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        # Build essentials
        build-essential \
        cmake \
        ninja-build \
        ccache \
        git \
        curl \
        wget \
        # Python development
        python3-dev \
        python3-venv \
        # GUI support for Deep Tree Echo
        python3-tk \
        xvfb \
        x11vnc \
        xauth \
        xdotool \
        fluxbox \
        # Browser automation support
        firefox-esr \
        chromium \
        chromium-driver \
        # Graphics and display
        libgtk-3-0 \
        libdbus-glib-1-2 \
        fonts-liberation \
        dbus \
        dbus-x11 \
        libx11-xcb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxfixes3 \
        libxi6 \
        libxtst6 \
        libxss1 \
        libxrandr2 \
        libasound2 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libatspi2.0-0 \
        libcups2 \
        libdrm2 \
        libgbm1 \
        libnss3 \
        libxshmfence1 \
        xdg-utils \
        x11-utils \
        # System utilities
        sudo \
        vim \
        nano \
        htop \
        tree \
        jq \
        unzip \
        zip \
        rsync \
        # Network tools
        net-tools \
        iputils-ping \
        # Development tools
        gdb \
        valgrind \
        strace \
        # OpenGL and CUDA prep
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libfontconfig1 \
        libxrender1 \
        libssl-dev \
        libcap2 \
        libtinfo5 \
        libaio-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up environment for GUI applications
ENV DISPLAY=:1
ENV XAUTHORITY=/tmp/.Xauthority
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix && \
    touch /tmp/.Xauthority && chmod 644 /tmp/.Xauthority && \
    mkdir -p /var/run/dbus && chmod 755 /var/run/dbus

# Install CUDA Toolkit (conditional, can be skipped for CPU-only)
ARG INSTALL_CUDA=false
ARG CUDA_VERSION=12.4

# CUDA installation block
RUN if [ "$INSTALL_CUDA" = "true" ] ; then \
        echo "Installing CUDA $CUDA_VERSION..." && \
        wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        apt-get update && \
        cuda_version_dashes=$(echo $CUDA_VERSION | tr "." "-") && \
        apt-get -y install cuda-${cuda_version_dashes} cuda-nvcc-${cuda_version_dashes} cuda-libraries-dev-${cuda_version_dashes} && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* && \
        rm cuda-keyring_1.1-1_all.deb ; \
    fi

# Set CUDA environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install ROCm (for AMD GPU support) - conditional
ARG INSTALL_ROCM=false
RUN if [ "$INSTALL_ROCM" = "true" ] ; then \
        echo "Installing ROCm..." && \
        wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
        echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ ubuntu main' | tee /etc/apt/sources.list.d/rocm.list && \
        apt-get update && \
        apt-get install -y rocm-dev rocm-utils && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* ; \
    fi

# Set ROCm environment variables
ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Upgrade pip and install uv for fast package management
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install uv

# Install Python build requirements
COPY requirements/build.txt /tmp/requirements-build.txt
RUN uv pip install --system -r /tmp/requirements-build.txt

# Install Python development requirements
COPY requirements/lint.txt /tmp/requirements-lint.txt
COPY requirements/dev.txt /tmp/requirements-dev.txt
RUN uv pip install --system -r /tmp/requirements-lint.txt || true

# Install exact versions for code quality tools
RUN uv pip install --system \
        ruff==0.1.5 \
        codespell==2.3.0 \
        isort==5.13.2 \
        mypy==1.7.1 \
        black==24.10.0 \
        pre-commit==4.0.1

# Install browser automation tools
RUN uv pip install --system \
        playwright==1.51.0 \
        selenium==4.15.2 \
        pyautogui==0.9.54 \
        pynput==1.8.1

# Install Playwright browsers
RUN mkdir -p /ms-playwright && \
    playwright install --with-deps firefox chromium webkit || echo "Browser installation completed with warnings"

# Create X server initialization script
COPY .devcontainer/init-xserver.sh /usr/local/bin/init-xserver.sh
RUN chmod +x /usr/local/bin/init-xserver.sh

# Create workspace directories with proper ownership
RUN mkdir -p /workspace && \
    mkdir -p /workspace/.ccache && \
    chown -R vscode:vscode /workspace

# Switch to vscode user for the rest of the setup
USER vscode

# Set up ccache for faster rebuilds
ENV CCACHE_DIR=/workspace/.ccache
ENV CCACHE_MAXSIZE=5G
ENV CC="ccache gcc"
ENV CXX="ccache g++"

# Create user directories
RUN mkdir -p /home/vscode/.local/bin && \
    mkdir -p /home/vscode/.cache && \
    mkdir -p /home/vscode/.config

# Set up development environment variables
ENV ECHO_ENABLE_DEEP_TREE=true
ENV ECHO_ENABLE_VM_DAEMON=true
ENV DEEP_TREE_ECHO_MODE=development
ENV CMAKE_BUILD_TYPE=Release
ENV MAX_JOBS=4

# Create development aliases and helpers
RUN echo 'alias ll="ls -la"' >> /home/vscode/.bashrc && \
    echo 'alias la="ls -la"' >> /home/vscode/.bashrc && \
    echo 'alias ..="cd .."' >> /home/vscode/.bashrc && \
    echo 'alias ...="cd ../.."' >> /home/vscode/.bashrc && \
    echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> /home/vscode/.bashrc && \
    echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> /home/vscode/.bashrc

# Add development helper functions
RUN echo 'function aphrodite-build() {' >> /home/vscode/.bashrc && \
    echo '    echo "Building Aphrodite Engine..."' >> /home/vscode/.bashrc && \
    echo '    export APHRODITE_TARGET_DEVICE=${APHRODITE_TARGET_DEVICE:-cpu}' >> /home/vscode/.bashrc && \
    echo '    echo "Target device: $APHRODITE_TARGET_DEVICE"' >> /home/vscode/.bashrc && \
    echo '    pip install -e . --timeout 36000' >> /home/vscode/.bashrc && \
    echo '}' >> /home/vscode/.bashrc && \
    echo 'function aphrodite-test() {' >> /home/vscode/.bashrc && \
    echo '    echo "Running Aphrodite tests..."' >> /home/vscode/.bashrc && \
    echo '    python -c "from aphrodite import LLM, SamplingParams; print(\"✅ Core imports successful\")"' >> /home/vscode/.bashrc && \
    echo '    aphrodite --help > /dev/null && echo "✅ CLI accessible" || echo "❌ CLI not accessible"' >> /home/vscode/.bashrc && \
    echo '}' >> /home/vscode/.bashrc && \
    echo 'function echo-status() {' >> /home/vscode/.bashrc && \
    echo '    echo "=== Deep Tree Echo Status ==="' >> /home/vscode/.bashrc && \
    echo '    echo "ECHO_ENABLE_DEEP_TREE: $ECHO_ENABLE_DEEP_TREE"' >> /home/vscode/.bashrc && \
    echo '    echo "ECHO_ENABLE_VM_DAEMON: $ECHO_ENABLE_VM_DAEMON"' >> /home/vscode/.bashrc && \
    echo '    echo "DEEP_TREE_ECHO_MODE: $DEEP_TREE_ECHO_MODE"' >> /home/vscode/.bashrc && \
    echo '    echo "APHRODITE_TARGET_DEVICE: $APHRODITE_TARGET_DEVICE"' >> /home/vscode/.bashrc && \
    echo '    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"' >> /home/vscode/.bashrc && \
    echo '}' >> /home/vscode/.bashrc

# Switch back to root for final setup
USER root

# Create ccache wrapper scripts
RUN mkdir -p /usr/local/bin && \
    echo '#!/bin/bash\nccache gcc "$@"' > /usr/local/bin/gcc && \
    echo '#!/bin/bash\nccache g++ "$@"' > /usr/local/bin/g++ && \
    chmod +x /usr/local/bin/gcc /usr/local/bin/g++

# Final permissions setup
RUN chown -R vscode:vscode /workspace /tmp/.X* /var/run/dbus || true

# Switch back to vscode user
USER vscode

WORKDIR /workspace

# Add labels for identification
LABEL maintainer="dtecho <dtecho@users.noreply.github.com>"
LABEL version="1.0"
LABEL description="Aphrodite Engine development container with Deep Tree Echo support"
LABEL target-devices="cpu,cuda,rocm,tpu,xpu"
LABEL features="gui,browser-automation,multi-arch,ccache"