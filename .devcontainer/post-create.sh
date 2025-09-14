#!/bin/bash
# Post-create script for Aphrodite Engine devcontainer
# Runs once after the container is created

set -e

echo "🚀 Starting Aphrodite Engine devcontainer post-create setup..."

# Update git configuration
echo "📝 Setting up git configuration..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main

# Initialize X server for GUI applications
echo "🖥️  Initializing X server..."
bash /usr/local/bin/init-xserver.sh || echo "X server setup completed with warnings"

# Set up ccache
echo "⚡ Setting up ccache for faster builds..."
ccache --set-config=max_size=5G
ccache --set-config=compression=true
ccache --set-config=hash_dir=false
ccache --zero-stats
echo "ccache configured with 5GB limit"

# Check system dependencies
echo "🔍 Verifying system dependencies..."
python3 --version
cmake --version
ninja --version
ccache --version

# Check CUDA availability (if installed)
if command -v nvcc &> /dev/null; then
    echo "🔥 CUDA detected:"
    nvcc --version
    export APHRODITE_TARGET_DEVICE=cuda
else
    echo "💻 Using CPU-only build"
    export APHRODITE_TARGET_DEVICE=cpu
fi

# Check ROCm availability (if installed)
if command -v rocm-smi &> /dev/null; then
    echo "🔶 ROCm detected:"
    rocm-smi --version || echo "ROCm tools available"
fi

# Install Python test dependencies
echo "📦 Installing Python test dependencies..."
if [ -f requirements/test.txt ]; then
    pip install -r requirements/test.txt --timeout 3600 || echo "Some test dependencies may have failed to install"
fi

# Install development pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
if [ -f .pre-commit-config.yaml ]; then
    pre-commit install || echo "Pre-commit hooks setup completed with warnings"
fi

# Create development workspace directories
echo "📁 Creating development directories..."
mkdir -p /workspace/logs
mkdir -p /workspace/models
mkdir -p /workspace/checkpoints
mkdir -p /workspace/experiments
mkdir -p /workspace/browser_data
mkdir -p /workspace/deep_tree_echo_profile

# Set up environment file
echo "⚙️  Creating development environment file..."
cat > /workspace/.env.development << EOF
# Aphrodite Engine Development Environment
APHRODITE_TARGET_DEVICE=${APHRODITE_TARGET_DEVICE}
ECHO_ENABLE_DEEP_TREE=true
ECHO_ENABLE_VM_DAEMON=true
DEEP_TREE_ECHO_MODE=development
CMAKE_BUILD_TYPE=Release
MAX_JOBS=4
CCACHE_DIR=/workspace/.ccache
PYTHONPATH=/workspace
PYTHONUNBUFFERED=1
HF_HUB_ENABLE_HF_TRANSFER=1

# Display settings for GUI apps
DISPLAY=:1
XAUTHORITY=/tmp/.Xauthority

# Development flags
DEBUG=true
VERBOSE=true
EOF

# Install browser dependencies for Deep Tree Echo
echo "🌐 Setting up browser automation..."
if command -v playwright &> /dev/null; then
    playwright install-deps || echo "Browser dependencies setup completed with warnings"
fi

# Create development helper scripts
echo "🛠️  Creating development helper scripts..."
cat > /workspace/quick-start.sh << 'EOF'
#!/bin/bash
# Quick start script for Aphrodite Engine development

echo "🚀 Aphrodite Engine Quick Start"
echo "================================"

# Load environment
source .env.development 2>/dev/null || true

echo "Environment:"
echo "  Target Device: $APHRODITE_TARGET_DEVICE"
echo "  Deep Tree Echo: $ECHO_ENABLE_DEEP_TREE"
echo "  VM Daemon: $ECHO_ENABLE_VM_DAEMON"
echo ""

echo "Available commands:"
echo "  ./quick-start.sh build    - Build Aphrodite Engine"
echo "  ./quick-start.sh test     - Run basic tests"
echo "  ./quick-start.sh lint     - Run code quality checks"
echo "  ./quick-start.sh serve    - Start API server"
echo "  ./quick-start.sh status   - Show system status"
echo ""

case $1 in
    build)
        echo "🏗️  Building Aphrodite Engine..."
        export APHRODITE_TARGET_DEVICE=${APHRODITE_TARGET_DEVICE:-cpu}
        pip install -e . --timeout 36000
        ;;
    test)
        echo "🧪 Running basic tests..."
        python -c "from aphrodite import LLM, SamplingParams; print('✅ Core imports successful')"
        aphrodite --help > /dev/null && echo "✅ CLI accessible" || echo "❌ CLI not accessible"
        ;;
    lint)
        echo "🔍 Running code quality checks..."
        ruff check . --show-source || true
        isort --check-only . || true
        mypy --follow-imports skip aphrodite/ || true
        ;;
    serve)
        echo "🌐 Starting API server..."
        echo "This will start the Aphrodite API server on port 2242"
        echo "Use a small model for testing, e.g.:"
        echo "aphrodite run microsoft/DialoGPT-medium --port 2242"
        ;;
    status)
        echo "📊 System Status:"
        echo "  Python: $(python3 --version)"
        echo "  CMake: $(cmake --version | head -1)"
        echo "  Ninja: $(ninja --version)"
        echo "  ccache: $(ccache --version | head -1)"
        echo "  ccache stats:"
        ccache --show-stats | head -10
        echo "  Disk usage:"
        df -h /workspace | tail -1
        ;;
    *)
        echo "Usage: $0 {build|test|lint|serve|status}"
        ;;
esac
EOF

chmod +x /workspace/quick-start.sh

# Set up Deep Tree Echo specific configuration
echo "🌳 Setting up Deep Tree Echo configuration..."
if [ -d "echo.kern" ]; then
    echo "Found echo.kern directory"
fi
if [ -d "echo.self" ]; then
    echo "Found echo.self directory"
fi
if [ -d "echo.dash" ]; then
    echo "Found echo.dash directory"
fi

# Create development documentation
echo "📚 Creating development documentation..."
cat > /workspace/DEVELOPMENT_README.md << 'EOF'
# Aphrodite Engine Development Container

This development container is optimized for Aphrodite Engine development with Deep Tree Echo integration.

## Quick Start

```bash
# Build the engine
./quick-start.sh build

# Run tests
./quick-start.sh test

# Start development server
./quick-start.sh serve
```

## Available Tools

- **Python 3.12** with all development dependencies
- **CMake 3.26+** for building C++ extensions
- **Ninja** for fast builds
- **ccache** for faster rebuilds (5GB cache)
- **ruff, mypy, isort, black** for code quality
- **Playwright, Selenium** for browser automation
- **X11 forwarding** for GUI applications

## Target Devices

Set the `APHRODITE_TARGET_DEVICE` environment variable:
- `cpu` - CPU-only build (default)
- `cuda` - NVIDIA GPU support
- `rocm` - AMD GPU support
- `tpu` - Google TPU support
- `xpu` - Intel XPU support

## Deep Tree Echo Features

- GUI application support via X11
- Browser automation capabilities
- Memory management systems
- VM daemon integration
- Development profiling tools

## Ports

- `2242` - Aphrodite API server
- `8000` - Alternative API port
- `6006` - TensorBoard
- `8888` - Jupyter notebooks
- `5901` - VNC display

## Helper Commands

```bash
# Check environment status
echo-status

# Build Aphrodite Engine
aphrodite-build

# Test installation
aphrodite-test

# Show ccache statistics
ccache --show-stats
```

## Troubleshooting

1. **Build fails**: Check disk space and memory usage
2. **GUI apps don't work**: Run `bash /usr/local/bin/init-xserver.sh`
3. **Slow builds**: Verify ccache is working with `ccache --show-stats`
4. **Import errors**: Check `PYTHONPATH=/workspace`

## Development Workflow

1. Make changes to the code
2. Run `ruff check . --fix` to fix linting issues
3. Run `./quick-start.sh test` to verify changes
4. Use `./quick-start.sh build` to rebuild if needed
5. Test with a small model using `./quick-start.sh serve`
EOF

echo "✅ Post-create setup complete!"
echo ""
echo "🎉 Aphrodite Engine development environment is ready!"
echo ""
echo "Quick start commands:"
echo "  ./quick-start.sh build    - Build the engine"
echo "  ./quick-start.sh test     - Run basic tests"
echo "  ./quick-start.sh status   - Show system status"
echo ""
echo "See DEVELOPMENT_README.md for detailed information."