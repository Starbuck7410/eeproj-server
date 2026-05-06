#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
echo "Starting Server..."
python3 server_processor.py
