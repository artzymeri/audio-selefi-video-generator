#!/bin/bash
# ============================================
# Setup Script for Islamic Reminder Video Bot
# Run this once: chmod +x setup.sh && ./setup.sh
# ============================================

set -e

echo "🕌 Islamic Reminder Video Bot - Setup"
echo "======================================"

# Check Python version
echo ""
echo "1. Checking Python..."
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    echo "   ✅ $PY_VERSION"
else
    echo "   ❌ Python 3 not found. Install it from https://www.python.org/downloads/"
    exit 1
fi

# Check FFmpeg
echo ""
echo "2. Checking FFmpeg..."
if command -v ffmpeg &>/dev/null; then
    FF_VERSION=$(ffmpeg -version 2>&1 | head -n1)
    echo "   ✅ $FF_VERSION"
else
    echo "   ❌ FFmpeg not found."
    echo "   Install with: brew install ffmpeg"
    echo "   Or download from: https://ffmpeg.org/download.html"
    exit 1
fi

# Create virtual environment
echo ""
echo "3. Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment already exists"
fi

# Activate and install
echo ""
echo "4. Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "   ✅ All packages installed"

# Create .env from example if it doesn't exist
echo ""
echo "5. Setting up .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   ✅ Created .env from template"
    echo "   ⚠️  IMPORTANT: Edit .env and add your API keys!"
else
    echo "   ✅ .env already exists"
fi

# Create assets placeholder
echo ""
echo "6. Checking assets..."
if [ ! -f "assets/logo.png" ]; then
    echo "   ⚠️  Place your logo.png in the assets/ folder"
else
    echo "   ✅ logo.png found"
fi

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Place logo.png in assets/"
echo "  3. Activate env: source venv/bin/activate"
echo "  4. Wait for Phase 2 code!"
echo "======================================"
