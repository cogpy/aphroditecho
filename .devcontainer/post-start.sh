#!/bin/bash
# Post-start script for Aphrodite Engine devcontainer
# Runs every time the container starts

set -e

echo "🔄 Starting Aphrodite Engine devcontainer post-start setup..."

# Ensure X server is running for GUI applications
echo "🖥️  Checking X server status..."
if [ ! -e /tmp/.X11-unix/X1 ]; then
    echo "Starting X server..."
    bash /usr/local/bin/init-xserver.sh || echo "X server setup completed with warnings"
else
    echo "✅ X server is already running"
fi

# Load development environment
if [ -f /workspace/.env.development ]; then
    echo "⚙️  Loading development environment..."
    set -a
    source /workspace/.env.development
    set +a
    echo "Environment loaded: target device = $APHRODITE_TARGET_DEVICE"
fi

# Check ccache status
echo "⚡ ccache status:"
ccache --show-stats | head -3

# Check disk space
echo "💾 Disk space:"
df -h /workspace | tail -1

# Show useful information
echo ""
echo "🎯 Development Environment Ready!"
echo "================================="
echo "Target device: $APHRODITE_TARGET_DEVICE"
echo "Workspace: /workspace"
echo "Python: $(python3 --version)"
echo ""
echo "Quick commands:"
echo "  ./quick-start.sh status  - Show detailed system status"
echo "  ./quick-start.sh build   - Build Aphrodite Engine"
echo "  ./quick-start.sh test    - Run basic tests"
echo ""
echo "Happy coding! 🚀"