#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
echo "Starting AI Server & Dashboard..."
python3 server_processor.py