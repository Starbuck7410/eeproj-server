# Autonomous Beach Cleaning System — Base Station

A sophisticated real-time computer vision and monitoring dashboard designed to track objects (balls, garbage, and baskets) using HSV color segmentation and AprilTag localization. The system features a robust location filtering mechanism to mitigate sensor noise and provides a web-based interface for live telemetry and video streaming.

## Architecture

The system is architected into several modular components working in concert:

*   **Vision Engine (`vision_core.py`):**
    *   **Color Detection:** Utilizes HSV (Hue, Saturation, Value) color space sampling of central pixels to classify detected circular objects.
    *   **AprilTag Processing:** Calculates precise coordinate centers from tag corner points for spatial localization using homography matrices.
*   **Processing Layer (`processing.py` & `main.py`):**
    *   Asynchronously handles video frame ingestion, connectivity status (Online/Offline), and triggers the detection algorithms.
    *   Manages the MJPEG video stream via Flask for real-time web viewing.
*   **Shared State Manager (`shared_state.py` & `location.py`):**
    *   **Thread-Safe Storage:** Uses `threading.Lock` to manage global access to frames, resolution, and object coordinates across different execution threads.
    *   **Spatial Filtering (`Location` class):** Implements a rolling buffer system to handle "jumps" in coordinate data. It uses robust outlier buffers with validation thresholds and rolling average smoothing to ensure that only stable, trusted coordinates are reported to the API.
*   **Web Dashboard (`main.py`, `templates/`):**
    *   A Flask-based web server providing:
        *   **Secure Login:** POC for session-managed access control.
        *   **Live Video Feed:** Real-time MJPEG stream of the processed camera feed.
        *   **Telemetry API:** JSON endpoints for retrieving real-time validated coordinates of the "garbage" and "basket" targets.

## Implementation Details

### Object Detection Logic
The system performs detection in two primary ways:
1.  **Circular Color Objects:** By sampling pixels within detected circles and comparing them against predefined `color_ranges`.
2.  **AprilTags:** Utilizing corner-to-center mean calculations to establish precise reference points for the environment.

### Robust Location Tracking
To prevent erratic movements in reported data (e.g., from lighting changes or occlusion), the `Location` module:
*   Maintains an `outlier_buffer`.
*   Requires a `persistence_threshold` of consecutive valid samples before validating a new location.
*   Uses an invalidation counter to reset/clear coordinates if tracking is lost, ensuring high data integrity for downstream tasks (like robot navigation).

## 🚀 Setup and Execution

### Prerequisites
*   Python 3
*   Required libraries: `opencv-python`, `numpy`, `flask`, `pyzmq` and `threading`.

### Installation & Running
1.  **Navigate to the project root:**
    ```bash
    cd path/to/vision_core
    ```
2.  **Initialize and run via the provided script:**
    The `start.sh` script automates environment activation and server startup.
    ```bash
    ./start.sh
    ```

## API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/video_feed` | `GET` | Returns the live MJPEG video stream. |
| `/api/garbage` | `GET` | Returns JSON: `{x, y, valid}` for garbage location. |
| `/api/basket` | `GET` | Returns JSON: `{x, y, valid}` for basket location. |
| `/login` | `POST` | Authenticates user credentials for the dashboard. |

## Project Structure
```text
├── config.py          # System thresholds and ID configurations
├── location.py        # Spatial tracking and filtering logic
├── main.py            # Flask web server and API endpoints
├── processing.py      # Core processing loop and stream handling
├── shared_state.py    # Global state and thread management
├── start.sh           # Startup automation script
├── vision_core.py     # Computer vision algorithms (HSV/AprilTag)
└── templates/         # HTML frontend (index, login)
```