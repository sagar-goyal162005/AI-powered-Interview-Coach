#!/bin/bash
# Render build script

set -e  # Exit on error

echo "Installing system dependencies..."
# Install system dependencies (ffmpeg for moviepy)
if [ -f packages.txt ]; then
    apt-get update
    xargs -a packages.txt apt-get install -y
fi

echo "Upgrading pip..."
# Install Python dependencies
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Build completed successfully!"
