#!/bin/bash
# Render build script

# Install system dependencies (ffmpeg for moviepy)
if [ -f packages.txt ]; then
    apt-get update
    xargs apt-get install -y < packages.txt
fi

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
