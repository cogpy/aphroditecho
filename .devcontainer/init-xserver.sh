#!/bin/bash
# X Server initialization script for GUI applications in devcontainer

set -e

echo "🖥️  Initializing X Server for GUI applications..."

# Kill any existing X server processes
pkill Xvfb || true
pkill x11vnc || true
pkill fluxbox || true
sleep 1

# Set up X authority
export DISPLAY=:1
export XAUTHORITY=/tmp/.Xauthority
touch $XAUTHORITY
chmod 644 $XAUTHORITY

# Start Xvfb (X Virtual Framebuffer)
echo "Starting Xvfb on display :1..."
Xvfb :1 -screen 0 1024x768x16 -ac -pn -noreset &
XVFB_PID=$!
echo "Xvfb started with PID: $XVFB_PID"

# Wait for X server to start
sleep 2

# Start window manager
echo "Starting fluxbox window manager..."
DISPLAY=:1 fluxbox &
FLUXBOX_PID=$!
echo "Fluxbox started with PID: $FLUXBOX_PID"

# Start VNC server for remote access
echo "Starting VNC server on port 5901..."
x11vnc -display :1 -rfbport 5901 -forever -shared -bg &
VNC_PID=$!
echo "VNC server started with PID: $VNC_PID"

# Test X server
echo "Testing X server..."
if DISPLAY=:1 xdpyinfo > /dev/null 2>&1; then
    echo "✅ X server is running successfully"
else
    echo "❌ X server test failed"
    exit 1
fi

echo "🖥️  X Server initialization complete!"
echo "   - Display: $DISPLAY"
echo "   - VNC Port: 5901"
echo "   - Resolution: 1024x768x16"
echo ""
echo "To test GUI applications, run:"
echo "  DISPLAY=:1 firefox &"
echo "  DISPLAY=:1 chromium &"