# OUTDATED README PLEASE DO NOT FOLLOW THESE INSTRUCTIONS

This server receives compressed video from a Raspberry Pi, performs YOLOv8 object detection and AprilTag 3D tracking, and hosts a web dashboard.

## Setup & Execution

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure IP:**
   - Note your Server's Local IP address
   - Ensure port `5555` and `5000` are open on your firewall

3. **Run the Server:**
   ```bash
   chmod +x start_server.sh
   ./start_server.sh
   ```

4. **View Stream:**
   - Open your browser and go to `http://<YOUR_SERVER_IP>:5000`

## Configuration
- **Confidence:** Adjust `CONFIDENCE_THRESHOLD` in `server_processor.py` to filter weak detections.